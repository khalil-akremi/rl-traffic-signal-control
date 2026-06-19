import sumo_rl
import numpy as np
import yaml
from pettingzoo import ParallelEnv
from environment.traffic_env import TrafficEnvironment


def run_fixed_timing(config_path='configs/config.yaml', episodes=5, phase_duration=30):
    """
    Fixed timing baseline.
    
    Each intersection cycles through phases with a fixed duration.
    No adaptation to traffic state whatsoever.
    This is what traditional traffic systems do.
    
    Args:
        config_path: path to config file
        episodes: number of evaluation episodes
        phase_duration: how many seconds each phase lasts (default 30s)
    """

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['environment']['use_gui'] = False
    delta_time = config['environment']['delta_time']

    print(f"\nRunning Fixed Timing Baseline")
    print(f"Phase duration: {phase_duration}s")
    print(f"Episodes: {episodes}")
    print("-" * 40)

    env = TrafficEnvironment(config['environment'])

    # how many steps before switching phase
    steps_per_phase = phase_duration // delta_time

    all_episode_rewards = []

    for episode in range(episodes):
        observations, infos = env.reset(seed=episode)

        episode_reward = 0
        step = 0
        done = False

        while not done:
            # fixed timing logic — purely time based, ignore observations
            # cycle phase 0 and 1 alternately based on step count
            current_phase = (step // steps_per_phase) % 2

            # same phase for all agents — fixed timing doesn't adapt
            actions_dict = {
                agent: current_phase
                for agent in env.possible_agents
            }

            observations, rewards, terminations, truncations, infos = \
                env.step(actions_dict)

            reward_values = [
                rewards.get(agent, 0.0)
                for agent in env.possible_agents
            ]
            episode_reward += np.mean(reward_values)
            step += 1

            done = (
                all(terminations.values()) or
                all(truncations.values())
            )

        all_episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: reward = {episode_reward:.2f}")

    avg_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)

    print(f"\nFixed Timing Results:")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Best episode:   {max(all_episode_rewards):.2f}")
    print(f"Worst episode:  {min(all_episode_rewards):.2f}")

    env.close()
    return avg_reward, all_episode_rewards


def run_random_baseline(config_path='configs/config.yaml', episodes=5):
    """
    Random agent baseline.
    
    Each agent picks a random action at every timestep.
    This is the absolute floor — if our model can't beat this,
    something is seriously wrong.
    """

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['environment']['use_gui'] = False

    print(f"\nRunning Random Baseline")
    print(f"Episodes: {episodes}")
    print("-" * 40)

    env = TrafficEnvironment(config['environment'])

    all_episode_rewards = []

    for episode in range(episodes):
        observations, infos = env.reset(seed=episode)

        episode_reward = 0
        done = False

        while not done:
            # completely random actions
            actions_dict = {
                agent: env.action_space(agent).sample()
                for agent in env.possible_agents
            }

            observations, rewards, terminations, truncations, infos = \
                env.step(actions_dict)

            reward_values = [
                rewards.get(agent, 0.0)
                for agent in env.possible_agents
            ]
            episode_reward += np.mean(reward_values)
            done = (
                all(terminations.values()) or
                all(truncations.values())
            )

        all_episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: reward = {episode_reward:.2f}")

    avg_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)

    print(f"\nRandom Baseline Results:")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Best episode:   {max(all_episode_rewards):.2f}")
    print(f"Worst episode:  {min(all_episode_rewards):.2f}")

    env.close()
    return avg_reward, all_episode_rewards


if __name__ == '__main__':

    print("=" * 50)
    print("BASELINE EVALUATION")
    print("=" * 50)

    # run both baselines
    fixed_avg, fixed_rewards = run_fixed_timing(episodes=5)
    random_avg, random_rewards = run_random_baseline(episodes=5)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Fixed Timing:  {fixed_avg:.2f}")
    print(f"Random Agent:  {random_avg:.2f}")
    print(f"GNN-MAPPO:     -291.95  (from training)")
    print("=" * 50)

    # quick comparison
    if -291.95 < fixed_avg:
        improvement = (fixed_avg - (-291.95)) / abs(fixed_avg) * 100
        print(f"\nGNN-MAPPO is {improvement:.1f}% better than fixed timing!")
    else:
        print(f"\nGNN-MAPPO needs more training to beat fixed timing.")