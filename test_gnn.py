import torch
import yaml
from environment.traffic_env import TrafficEnvironment
from models.gat_encoder import GATEncoder, GraphBuilder

# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# build environment
env = TrafficEnvironment(config['environment'])
observations, infos = env.reset(seed=42)

# build graph structure
edges_src, edges_dst, agent_to_idx = env.get_graph_structure()
edge_pairs = list(zip(edges_src, edges_dst))

# build graph builder
graph_builder = GraphBuilder(
    edge_index=edge_pairs,
    num_nodes=env._num_agents,
    device=device
)

# build GNN config
gnn_config = {
    'obs_size': env.obs_size,
    **config['model']
}

# build encoder
encoder = GATEncoder(gnn_config).to(device)
print(f"\nGAT Encoder architecture:")
print(encoder)

# count parameters
total_params = sum(p.numel() for p in encoder.parameters())
print(f"\nTotal trainable parameters: {total_params:,}")

# forward pass
node_features, edge_index = graph_builder.build(
    observations,
    env.possible_agents
)

print(f"\nInput node features shape: {node_features.shape}")
print(f"Edge index shape: {edge_index.shape}")

# run through encoder
embeddings = encoder(node_features, edge_index)
print(f"Output embeddings shape: {embeddings.shape}")

# test with attention weights
embeddings, attention = encoder(
    node_features, edge_index, return_attention=True
)
edge_idx, attn_w1, attn_w2 = attention
print(f"\nAttention weights layer 1 shape: {attn_w1.shape}")
print(f"Attention weights layer 2 shape: {attn_w2.shape}")
print(f"Sample attention weights: {attn_w1[:5]}")

env.close()
print("\nGNN test passed!")