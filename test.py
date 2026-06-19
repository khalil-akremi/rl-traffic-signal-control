import sumo_rl
import yaml

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

from environment.traffic_env import TrafficEnvironment
env = TrafficEnvironment(config['environment'])
obs, infos = env.reset()

actions = {agent: 0 for agent in env.possible_agents}
for step in range(5):
    obs, rewards, terms, truncs, infos = env.step(actions)
    print(f"\nStep {step}:")
    for agent in env.possible_agents:
        print(f"  {agent}: phase = {obs[agent][:2]}")

env.close()