import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class Actor(nn.Module):
    def __init__(self, embedding_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

        self._init_weights()

    def _init_weights(self):
        layers = [l for l in self.network if isinstance(l, nn.Linear)]
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                is_last = (layer is layers[-1])
                gain = 0.01 if is_last else np.sqrt(2)
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

    def forward(self, embedding):
        logits = self.network(embedding)
        dist = Categorical(logits=logits)
        return dist, logits


class Critic(nn.Module):
    def __init__(self, num_agents, embedding_dim, hidden_dim=256):
        super(Critic, self).__init__()

        global_state_dim = num_agents * embedding_dim

        self.network = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def forward(self, all_embeddings):
        global_state = all_embeddings.flatten()
        value = self.network(global_state)
        return value


class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.embeddings = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, obs, embeddings, actions, log_probs, rewards, value, done):
        self.observations.append(obs)
        self.embeddings.append(embeddings.detach().cpu())
        self.actions.append(actions)
        self.log_probs.append(log_probs.detach().cpu())
        self.rewards.append(rewards)
        self.values.append(value.detach().cpu())
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


class MAPPO(nn.Module):
    def __init__(self, config, num_agents, device):
        super(MAPPO, self).__init__()

        self.num_agents = num_agents
        self.device = device

        self.gamma = config['gamma']
        self.clip_epsilon = config['clip_epsilon']
        self.update_epochs = config['update_epochs']
        self.batch_size = config['batch_size']
        self.lr_actor = config['lr_actor']
        self.lr_critic = config['lr_critic']
        # entropy annealing parameters
        self.entropy_coef_start = config.get('entropy_coef_start', 0.1)
        self.entropy_coef_end = config.get('entropy_coef_end', 0.005)
        self.entropy_anneal_episodes = config.get('entropy_anneal_episodes', 300)
        self.entropy_coef = self.entropy_coef_start  # current value, updated each episode
        self.value_coef = config.get('value_coef', 0.5)

        embedding_dim = config['embedding_dim']
        action_dim = config['action_dim']

        self.actor = Actor(
            embedding_dim=embedding_dim,
            action_dim=action_dim
        ).to(device)

        self.critic = Critic(
            num_agents=num_agents,
            embedding_dim=embedding_dim
        ).to(device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr_actor,
            eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.lr_critic,
            eps=1e-5
        )

        self.buffer = RolloutBuffer()

    def select_actions(self, embeddings):
        with torch.no_grad():
            dist, _ = self.actor(embeddings)
            actions_tensor = dist.sample()
            log_probs = dist.log_prob(actions_tensor)
            value = self.critic(embeddings)

        return actions_tensor, log_probs, value

    def compute_returns(self, rewards, values, dones, last_value):
        """
        Compute per-agent discounted returns using GAE.

        Key change from original:
            rewards[t] is a list of per-agent rewards, shape (num_agents,)
            We now compute one advantage per timestep using mean reward
            for the centralized critic, but keep per-agent rewards tracked
            so the critic learns from the true global signal while actors
            receive differentiated gradients through their own embeddings.

        Args:
            rewards: list of lists, each inner list shape (num_agents,)
            values: list of value tensors (scalars from centralized critic)
            dones: list of done flags
            last_value: value estimate for the final state

        Returns:
            returns: shape (T,) — one return per timestep for critic
            advantages: shape (T, num_agents) — per-agent advantages
        """
        gae_lambda = 0.95
        T = len(rewards)

        # convert rewards to tensor: shape (T, num_agents)
        rewards_tensor = torch.tensor(
            rewards, dtype=torch.float32
        )  # (T, num_agents)

        # per-agent returns and advantages
        returns = []
        advantages = []

        gae = torch.zeros(self.num_agents)  # one GAE per agent
        next_value = last_value

        for t in reversed(range(T)):
            # per-agent reward at timestep t
            reward_t = rewards_tensor[t]          # (num_agents,)
            done = dones[t]

            # centralized critic value (scalar) used as baseline
            value = values[t].item()

            # TD error per agent
            # critic gives one value for global state,
            # but each agent's reward is local
            delta = reward_t + self.gamma * next_value * (1 - done) - value

            # per-agent GAE
            gae = delta + self.gamma * gae_lambda * (1 - done) * gae

            returns.insert(0, gae + value)        # (num_agents,)
            advantages.insert(0, gae.clone())     # (num_agents,)

            next_value = value

        # returns: shape (T, num_agents)
        returns = torch.stack(returns).to(self.device)

        # advantages: shape (T, num_agents)
        advantages = torch.stack(advantages).to(self.device)

        # normalize advantages across all agents and timesteps
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        # critic trains on mean return across agents per timestep
        # shape (T,)
        returns_for_critic = returns.mean(dim=1)

        returns_for_critic = (returns_for_critic - returns_for_critic.mean()) / (returns_for_critic.std() + 1e-8)

        return returns_for_critic, advantages

    def update(self, embeddings_list, actions_list,
               log_probs_list, returns, advantages):
        """
        PPO update with per-agent advantages.

        Args:
            embeddings_list: list of (num_agents, embedding_dim) tensors
            actions_list: list of (num_agents,) tensors
            log_probs_list: list of (num_agents,) tensors
            returns: shape (T,) — critic targets
            advantages: shape (T, num_agents) — per-agent advantages
        """
        old_embeddings = torch.stack(embeddings_list).to(self.device)
        old_actions = torch.stack(actions_list).to(self.device)
        old_log_probs = torch.stack(log_probs_list).to(self.device)

        T = old_embeddings.shape[0]

        actor_losses = []
        critic_losses = []
        entropy_losses = []

        for epoch in range(self.update_epochs):

            # --- Actor loss ---
            embeddings_flat = old_embeddings.view(-1, old_embeddings.shape[-1])
            actions_flat = old_actions.view(-1)
            old_log_probs_flat = old_log_probs.view(-1)

            dist, _ = self.actor(embeddings_flat)
            new_log_probs = dist.log_prob(actions_flat)
            entropy = dist.entropy().mean()

            # advantages: (T, num_agents) → flatten to (T*num_agents,)
            # each agent gets its OWN advantage — this is the key fix
            adv_flat = advantages.reshape(-1)

            ratio = torch.exp(new_log_probs - old_log_probs_flat)
            surr1 = ratio * adv_flat
            surr2 = torch.clamp(
                ratio,
                1 - self.clip_epsilon,
                1 + self.clip_epsilon
            ) * adv_flat

            actor_loss = -torch.min(surr1, surr2).mean()

            # --- Critic loss (separate backward pass) ---
            # detach embeddings so critic update doesn't affect actor graph
            values_pred = torch.stack([
                self.critic(old_embeddings[t].detach()).squeeze()
                for t in range(T)
            ])
            critic_loss = F.mse_loss(values_pred, returns)

            # --- Actor update ---
            actor_total = actor_loss - self.entropy_coef * entropy
            self.actor_optimizer.zero_grad()
            actor_total.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            # --- Critic update (separate) ---
            self.critic_optimizer.zero_grad()
            (self.value_coef * critic_loss).backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy.item())

        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropy_losses)
        }
    def update_entropy_coef(self, episode):
        """
        Linearly anneal entropy coefficient from start to end
        over entropy_anneal_episodes episodes.

        After anneal period, stays at entropy_coef_end.

        Call this once per episode in the training loop.
        """
        if episode >= self.entropy_anneal_episodes:
            self.entropy_coef = self.entropy_coef_end
        else:
            # linear interpolation
            fraction = episode / self.entropy_anneal_episodes
            self.entropy_coef = (
                self.entropy_coef_start
                + fraction * (self.entropy_coef_end - self.entropy_coef_start)
            )