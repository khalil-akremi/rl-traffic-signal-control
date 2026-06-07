import yaml
from environment.traffic_env import TrafficEnvironment

with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

env = TrafficEnvironment(config['environment'])
observations, infos = env.reset(seed=42)

print("\n--- Padded observations ---")
for agent, obs in observations.items():
    print(f"{agent}: shape = {obs.shape}")

# test graph structure
edges_src, edges_dst, agent_to_idx = env.get_graph_structure()
print(f"\n--- Graph structure ---")
print(f"Number of nodes: {len(env.possible_agents)}")
print(f"Number of edges: {len(edges_src)}")
print(f"Agent to index mapping: {agent_to_idx}")
print(f"Edge list (first 10): {list(zip(edges_src, edges_dst))[:10]}")

env.close()
print("\nEnvironment test passed!")