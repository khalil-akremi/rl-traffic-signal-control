import torch
import numpy as np
import yaml
import wandb
import os
from collections import deque
from torch.amp import autocast, GradScaler

from environment.traffic_env import TrafficEnvironment
from models.gat_encoder import GATEncoder, MLPEncoder, GraphBuilder
from models.mappo import MAPPO


def train(config_path=None, override_config=None, trial_name=None):
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs', 'config.yaml'
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if override_config is not None:
        for key, value in override_config.items():
            section, param = key.split('.')
            config[section][param] = value

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print("\nInitializing environment...")
    env = TrafficEnvironment(config['environment'])
    env.set_train_mode()

    edges_src, edges_dst, agent_to_idx = env.get_graph_structure()
    edge_pairs = list(zip(edges_src, edges_dst))
    graph_builder = GraphBuilder(edge_pairs, env._num_agents, device)

    print("Building models...")
    gnn_config = {
        'obs_size': env.obs_size,
        **config['model']
    }

    # --- Model type switch: GAT vs plain MLP (no graph) ---
    model_type = config['model'].get('model_type', 'gat')
    if model_type == 'gat':
        encoder = GATEncoder(gnn_config).to(device)
        print("Using GATEncoder (with graph attention)")
    elif model_type == 'mlp':
        encoder = MLPEncoder(gnn_config).to(device)
        print("Using MLPEncoder (NO graph — ablation baseline)")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    mappo_config = {
        **config['mappo'],
        'embedding_dim': config['model']['embedding_dim'],
        'action_dim': 2
    }
    mappo = MAPPO(mappo_config, env._num_agents, device)

    joint_actor_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(mappo.actor.parameters()),
        lr=mappo_config['lr_actor'],
        eps=1e-5
    )
    mappo.actor_optimizer = joint_actor_optimizer

    scaler = GradScaler('cuda')

    start_episode = 0
    best_reward = -np.inf

    # checkpoint paths depend on model_type so GAT and MLP runs don't overwrite each other
    checkpoint_dir = f"checkpoints_{model_type}"

    if override_config is None:
        checkpoint_path = f'{checkpoint_dir}/latest_checkpoint.pt'
        if os.path.exists(checkpoint_path):
            print(f"\nFound checkpoint — resuming training...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            mappo.actor.load_state_dict(checkpoint['actor_state_dict'])
            mappo.critic.load_state_dict(checkpoint['critic_state_dict'])
            start_episode = checkpoint['episode'] + 1
            best_reward = checkpoint['best_reward']
            print(f"Resumed from episode {start_episode}")
            print(f"Best reward so far: {best_reward:.2f}")
        else:
            print("\nNo checkpoint found — starting fresh...")

    run_name = trial_name if trial_name else f"gnn-mappo-4x4-{model_type}-ep{start_episode}"
    wandb.init(
        project="traffic-marl-gnn",
        name=run_name,
        config=config,
        resume="allow" if override_config is None else None,
        reinit=True
    )

    episode_rewards = deque(maxlen=50)
    num_episodes = config['training']['episodes']
    update_freq = config['training'].get('update_freq', 256)

    encoder_params = sum(p.numel() for p in encoder.parameters())
    actor_params = sum(p.numel() for p in mappo.actor.parameters())
    critic_params = sum(p.numel() for p in mappo.critic.parameters())

    print(f"\n{'='*50}")
    print(f"Starting training — model_type: {model_type}")
    print(f"Episodes: {start_episode} → {num_episodes}")
    print(f"Agents: {env._num_agents}")
    print(f"Update frequency: {update_freq} steps")
    print(f"Encoder params: {encoder_params:,}")
    print(f"Actor params:   {actor_params:,}")
    print(f"Critic params:  {critic_params:,}")
    print(f"{'='*50}\n")

    for episode in range(start_episode, num_episodes):

        observations, infos = env.reset(seed=seed + episode)

        episode_reward = 0
        episode_steps = 0
        losses = {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}

        embeddings_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        values_list = []
        dones_list = []

        last_node_features = None
        last_edge_index = None
        last_node_ids = None

        done = False
        while not done:

            node_features, edge_index, node_ids = graph_builder.build(
                observations, env.possible_agents
            )
            last_node_features = node_features
            last_edge_index = edge_index
            last_node_ids = node_ids

            with autocast('cuda'):
                # MLPEncoder ignores edge_index internally, GATEncoder uses it
                embeddings = encoder(node_features, edge_index, node_ids)

            actions_tensor, log_probs, value = mappo.select_actions(
                embeddings.float()
            )

            actions_dict = {
                agent: actions_tensor[i].item()
                for i, agent in enumerate(env.possible_agents)
            }

            next_obs, rewards, terminations, truncations, infos = \
                env.step(actions_dict)

            reward_values = [
                rewards.get(agent, 0.0)
                for agent in env.possible_agents
            ]
            mean_reward = np.mean(reward_values)

            done = (
                all(terminations.values()) or
                all(truncations.values())
            )

            embeddings_list.append(embeddings.detach().cpu().float())
            actions_list.append(actions_tensor.detach().cpu())
            log_probs_list.append(log_probs.detach().cpu())
            rewards_list.append(reward_values)
            values_list.append(value.detach().cpu())
            dones_list.append(float(done))

            episode_reward += mean_reward
            episode_steps += 1
            observations = next_obs

            if len(embeddings_list) >= update_freq:
                losses = _update_models(
                    mappo, encoder, device,
                    embeddings_list, actions_list,
                    log_probs_list, rewards_list,
                    values_list, dones_list,
                    last_node_features, last_edge_index, last_node_ids
                )
                embeddings_list.clear()
                actions_list.clear()
                log_probs_list.clear()
                rewards_list.clear()
                values_list.clear()
                dones_list.clear()

        if len(embeddings_list) > 0:
            losses = _update_models(
                mappo, encoder, device,
                embeddings_list, actions_list,
                log_probs_list, rewards_list,
                values_list, dones_list,
                last_node_features, last_edge_index, last_node_ids
            )

        mappo.update_entropy_coef(episode)

        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)

        gpu_mem = 0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**2

        wandb.log({
            'episode': episode,
            'episode_reward': episode_reward,
            'avg_reward_50ep': avg_reward,
            'actor_loss': losses['actor_loss'],
            'critic_loss': losses['critic_loss'],
            'entropy': losses['entropy'],
            'entropy_coef': mappo.entropy_coef,
            'episode_steps': episode_steps,
            'gpu_memory_mb': gpu_mem,
            'best_reward': best_reward
        })

        if episode % 10 == 0:
            print(
                f"Ep {episode:4d} | "
                f"Reward: {episode_reward:9.2f} | "
                f"Avg(50): {avg_reward:9.2f} | "
                f"Loss A: {losses['actor_loss']:7.4f} | "
                f"Loss C: {losses['critic_loss']:7.4f} | "
                f"Entropy: {losses['entropy']:.4f} | "
                f"GPU: {gpu_mem:.0f}MB"
            )

        if override_config is None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            if episode % 10 == 0:
                _save_checkpoint(
                    encoder, mappo,
                    episode, avg_reward, best_reward,
                    checkpoint_dir, is_best=False
                )
            if avg_reward > best_reward and episode > 10:
                best_reward = avg_reward
                _save_checkpoint(
                    encoder, mappo,
                    episode, avg_reward, best_reward,
                    checkpoint_dir, is_best=True
                )
        else:
            if avg_reward > best_reward:
                best_reward = avg_reward

    print(f"\nTraining complete! Best avg reward: {best_reward:.2f}")

    env.close()
    wandb.finish()

    return best_reward


def _update_models(mappo, encoder, device,
                   embeddings_list, actions_list,
                   log_probs_list, rewards_list,
                   values_list, dones_list,
                   last_node_features, last_edge_index, last_node_ids):

    with torch.no_grad():
        last_embeddings = encoder(
            last_node_features, last_edge_index, last_node_ids
        ).float()
        last_value = mappo.critic(last_embeddings).item()

    returns, advantages = mappo.compute_returns(
        rewards_list, values_list, dones_list, last_value
    )

    losses = mappo.update(
        embeddings_list, actions_list,
        log_probs_list, returns, advantages
    )

    return losses


def _save_checkpoint(encoder, mappo, episode,
                     avg_reward, best_reward, checkpoint_dir, is_best=False):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'episode': episode,
        'avg_reward': avg_reward,
        'best_reward': best_reward,
        'encoder_state_dict': encoder.state_dict(),
        'actor_state_dict': mappo.actor.state_dict(),
        'critic_state_dict': mappo.critic.state_dict(),
    }

    torch.save(checkpoint, f'{checkpoint_dir}/latest_checkpoint.pt')

    if is_best:
        torch.save(checkpoint, f'{checkpoint_dir}/best_model.pt')
        print(f"  → New best model saved! Avg reward: {avg_reward:.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()
    train(config_path=args.config)


if __name__ == '__main__':
    main()