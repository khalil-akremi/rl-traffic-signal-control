import torch
import yaml
from environment.traffic_env import TrafficEnvironment
from models.gat_encoder import GATEncoder, GraphBuilder
from models.mappo import MAPPO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# build environment
env = TrafficEnvironment(config['environment'])
observations, infos = env.reset(seed=42)

# build graph
edges_src, edges_dst, agent_to_idx = env.get_graph_structure()
edge_pairs = list(zip(edges_src, edges_dst))
graph_builder = GraphBuilder(edge_pairs, env._num_agents, device)

# build models
gnn_config = {'obs_size': env.obs_size, **config['model']}
encoder = GATEncoder(gnn_config).to(device)

mappo_config = {
    **config['mappo'],
    'embedding_dim': config['model']['embedding_dim'],
    'action_dim': 2
}
mappo = MAPPO(mappo_config, env._num_agents, device)

print(f"\nActor parameters: {sum(p.numel() for p in mappo.actor.parameters()):,}")
print(f"Critic parameters: {sum(p.numel() for p in mappo.critic.parameters()):,}")

# test action selection
node_features, edge_index = graph_builder.build(observations, env.possible_agents)
embeddings = encoder(node_features, edge_index)

actions_tensor, log_probs, value = mappo.select_actions(embeddings)

print(f"\nEmbeddings shape: {embeddings.shape}")
print(f"Actions: {actions_tensor}")
print(f"Log probs: {log_probs}")
print(f"Value estimate: {value.item():.4f}")

# convert to dict for environment
actions_dict = {
    agent: actions_tensor[i].item()
    for i, agent in enumerate(env.possible_agents)
}
print(f"\nActions dict: {actions_dict}")

env.close()
print("\nMAPPO test passed!")