import torch
import yaml
import numpy as np
import traci
from collections import defaultdict
from environment.traffic_env import TrafficEnvironment
from models.gat_encoder import GATEncoder, MLPEncoder, GraphBuilder
from models.mappo import MAPPO


def get_traffic_metrics(env):
    try:
        if hasattr(env, '_sumo_env'):
            if hasattr(env._sumo_env, '_conn'):
                sumo = env._sumo_env._conn
            elif hasattr(env._sumo_env, 'sumo'):
                sumo = env._sumo_env.sumo
            else:
                sumo = traci
        else:
            sumo = traci
    except:
        sumo = traci

    try:
        all_vehicles = sumo.vehicle.getIDList()
    except:
        all_vehicles = traci.vehicle.getIDList()

    if len(all_vehicles) == 0:
        return {
            'avg_waiting_time': 0.0,
            'avg_queue_length': 0.0,
            'avg_speed': 0.0,
            'num_vehicles': 0
        }

    waiting_times = []
    speeds = []

    for veh in all_vehicles:
        try:
            waiting_times.append(sumo.vehicle.getAccumulatedWaitingTime(veh))
            speeds.append(sumo.vehicle.getSpeed(veh))
        except:
            waiting_times.append(traci.vehicle.getAccumulatedWaitingTime(veh))
            speeds.append(traci.vehicle.getSpeed(veh))

    queue_length = sum(1 for s in speeds if s < 0.1)

    return {
        'avg_waiting_time': np.mean(waiting_times),
        'avg_queue_length': queue_length,
        'avg_speed': np.mean(speeds),
        'num_vehicles': len(all_vehicles)
    }


def evaluate_model(
    config_path='configs/config.yaml',
    checkpoint_path=None,
    model_type='gat',
    episodes=30,
    use_gui=False,
    catastrophic_queue_threshold=50
):
    """
    catastrophic_queue_threshold: an episode is flagged as catastrophic
    if its average queue length exceeds this value. Used to quantify
    failure rate, not just average performance.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['environment']['use_gui'] = use_gui
    config['environment']['num_seconds'] = 3600
    config['model']['model_type'] = model_type

    if checkpoint_path is None:
        checkpoint_path = f'checkpoints_{model_type}/best_model.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = TrafficEnvironment(config['environment'])
    env.set_eval_mode()

    edges_src, edges_dst, _ = env.get_graph_structure()
    edge_pairs = list(zip(edges_src, edges_dst))
    graph_builder = GraphBuilder(edge_pairs, env._num_agents, device)

    gnn_config = {
        'obs_size': env.obs_size,
        **config['model']
    }

    if model_type == 'gat':
        encoder = GATEncoder(gnn_config).to(device)
    elif model_type == 'mlp':
        encoder = MLPEncoder(gnn_config).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    mappo_config = {
        **config['mappo'],
        'embedding_dim': config['model']['embedding_dim'],
        'action_dim': 2
    }
    mappo = MAPPO(mappo_config, env._num_agents, device)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    mappo.actor.load_state_dict(checkpoint['actor_state_dict'])
    print(f"Loaded from episode {checkpoint['episode']}")
    print(f"Training avg reward: {checkpoint['avg_reward']:.2f}")

    encoder.eval()
    mappo.actor.eval()

    print(f"\nRunning {episodes} evaluation episodes ({model_type.upper()})...")
    print("-" * 60)

    all_rewards = []
    all_waiting_times = []
    all_queue_lengths = []
    all_speeds = []
    catastrophic_count = 0

    for episode in range(episodes):
        observations, infos = env.reset(seed=200 + episode)

        episode_reward = 0
        episode_metrics = defaultdict(list)
        done = False
        step = 0

        while not done:
            node_features, edge_index, node_ids = graph_builder.build(
                observations, env.possible_agents
            )

            with torch.no_grad():
                embeddings = encoder(node_features, edge_index, node_ids)
                dist, _ = mappo.actor(embeddings.float())
                actions_tensor = dist.probs.argmax(dim=-1)

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
            episode_reward += np.mean(reward_values)

            if step % 10 == 0:
                try:
                    metrics = get_traffic_metrics(env)
                    episode_metrics['waiting_time'].append(
                        metrics['avg_waiting_time']
                    )
                    episode_metrics['queue_length'].append(
                        metrics['avg_queue_length']
                    )
                    episode_metrics['speed'].append(metrics['avg_speed'])
                except Exception as e:
                    pass

            step += 1
            observations = next_obs
            done = (
                all(terminations.values()) or
                all(truncations.values())
            )

        if episode_metrics['waiting_time']:
            avg_wait = np.mean(episode_metrics['waiting_time'])
            avg_queue = np.mean(episode_metrics['queue_length'])
            avg_speed = np.mean(episode_metrics['speed'])
        else:
            avg_wait = avg_queue = avg_speed = 0.0

        is_catastrophic = avg_queue > catastrophic_queue_threshold
        if is_catastrophic:
            catastrophic_count += 1

        all_rewards.append(episode_reward)
        all_waiting_times.append(avg_wait)
        all_queue_lengths.append(avg_queue)
        all_speeds.append(avg_speed)

        flag = " [CATASTROPHIC]" if is_catastrophic else ""
        print(
            f"Ep {episode+1:2d} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Wait: {avg_wait:6.1f}s | "
            f"Queue: {avg_queue:5.1f} veh | "
            f"Speed: {avg_speed:4.2f} m/s"
            f"{flag}"
        )

    env.close()

    failure_rate = (catastrophic_count / episodes) * 100

    print(f"\n{'='*60}")
    print(f"{model_type.upper()} EVALUATION RESULTS ({episodes} episodes)")
    print(f"{'='*60}")
    print(f"Reward:       {np.mean(all_rewards):8.2f} ± {np.std(all_rewards):.2f}")
    print(f"Waiting time: {np.mean(all_waiting_times):8.2f} ± {np.std(all_waiting_times):.2f} s")
    print(f"Queue length: {np.mean(all_queue_lengths):8.2f} ± {np.std(all_queue_lengths):.2f} veh")
    print(f"Avg speed:    {np.mean(all_speeds):8.2f} ± {np.std(all_speeds):.2f} m/s")
    print(f"Catastrophic failures: {catastrophic_count}/{episodes} ({failure_rate:.1f}%)")
    print(f"{'='*60}")

    # also compute median, which is robust to the catastrophic outliers
    print(f"\nMedian (robust to outliers):")
    print(f"Reward:       {np.median(all_rewards):8.2f}")
    print(f"Waiting time: {np.median(all_waiting_times):8.2f} s")
    print(f"Queue length: {np.median(all_queue_lengths):8.2f} veh")

    return {
        'reward': (np.mean(all_rewards), np.std(all_rewards)),
        'waiting_time': (np.mean(all_waiting_times), np.std(all_waiting_times)),
        'queue_length': (np.mean(all_queue_lengths), np.std(all_queue_lengths)),
        'speed': (np.mean(all_speeds), np.std(all_speeds)),
        'failure_rate': failure_rate,
        'median_reward': np.median(all_rewards)
    }


def evaluate_fixed_timing(
    config_path='configs/config.yaml',
    episodes=30,
    phase_duration=30,
    catastrophic_queue_threshold=50
):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['environment']['use_gui'] = False
    delta_time = config['environment']['delta_time']
    steps_per_phase = phase_duration // delta_time

    env = TrafficEnvironment(config['environment'])
    env.set_eval_mode()

    print(f"\nRunning Fixed Timing ({phase_duration}s phases)...")
    print("-" * 60)

    all_rewards = []
    all_waiting_times = []
    all_queue_lengths = []
    all_speeds = []
    catastrophic_count = 0

    for episode in range(episodes):
        observations, infos = env.reset(seed=200 + episode)

        episode_reward = 0
        episode_metrics = defaultdict(list)
        step = 0
        done = False

        while not done:
            current_phase = (step // steps_per_phase) % 2
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

            if step % 10 == 0:
                try:
                    metrics = get_traffic_metrics(env)
                    episode_metrics['waiting_time'].append(
                        metrics['avg_waiting_time']
                    )
                    episode_metrics['queue_length'].append(
                        metrics['avg_queue_length']
                    )
                    episode_metrics['speed'].append(metrics['avg_speed'])
                except Exception as e:
                    pass

            step += 1
            done = (
                all(terminations.values()) or
                all(truncations.values())
            )

        if episode_metrics['waiting_time']:
            avg_wait = np.mean(episode_metrics['waiting_time'])
            avg_queue = np.mean(episode_metrics['queue_length'])
            avg_speed = np.mean(episode_metrics['speed'])
        else:
            avg_wait = avg_queue = avg_speed = 0.0

        is_catastrophic = avg_queue > catastrophic_queue_threshold
        if is_catastrophic:
            catastrophic_count += 1

        all_rewards.append(episode_reward)
        all_waiting_times.append(avg_wait)
        all_queue_lengths.append(avg_queue)
        all_speeds.append(avg_speed)

        flag = " [CATASTROPHIC]" if is_catastrophic else ""
        print(
            f"Ep {episode+1:2d} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Wait: {avg_wait:6.1f}s | "
            f"Queue: {avg_queue:5.1f} veh | "
            f"Speed: {avg_speed:4.2f} m/s"
            f"{flag}"
        )

    env.close()

    failure_rate = (catastrophic_count / episodes) * 100

    print(f"\n{'='*60}")
    print(f"FIXED TIMING ({phase_duration}s) RESULTS ({episodes} episodes)")
    print(f"{'='*60}")
    print(f"Reward:       {np.mean(all_rewards):8.2f} ± {np.std(all_rewards):.2f}")
    print(f"Waiting time: {np.mean(all_waiting_times):8.2f} ± {np.std(all_waiting_times):.2f} s")
    print(f"Queue length: {np.mean(all_queue_lengths):8.2f} ± {np.std(all_queue_lengths):.2f} veh")
    print(f"Avg speed:    {np.mean(all_speeds):8.2f} ± {np.std(all_speeds):.2f} m/s")
    print(f"Catastrophic failures: {catastrophic_count}/{episodes} ({failure_rate:.1f}%)")
    print(f"{'='*60}")

    print(f"\nMedian (robust to outliers):")
    print(f"Reward:       {np.median(all_rewards):8.2f}")
    print(f"Waiting time: {np.median(all_waiting_times):8.2f} s")
    print(f"Queue length: {np.median(all_queue_lengths):8.2f} veh")

    return {
        'reward': (np.mean(all_rewards), np.std(all_rewards)),
        'waiting_time': (np.mean(all_waiting_times), np.std(all_waiting_times)),
        'queue_length': (np.mean(all_queue_lengths), np.std(all_queue_lengths)),
        'speed': (np.mean(all_speeds), np.std(all_speeds)),
        'failure_rate': failure_rate,
        'median_reward': np.median(all_rewards)
    }


if __name__ == '__main__':
    print("=" * 65)
    print("FULL COMPARISON: Fixed Timing vs MAPPO (no GAT) vs MAPPO+GAT")
    print("30 episodes each for statistical confidence")
    print("=" * 65)

    results_fixed = evaluate_fixed_timing(
        config_path='configs/config.yaml',
        episodes=30,
        phase_duration=10
    )

    results_mlp = evaluate_model(
        config_path='configs/config.yaml',
        model_type='mlp',
        episodes=30
    )

    results_gat = evaluate_model(
        config_path='configs/config.yaml',
        model_type='gat',
        episodes=30
    )

    print("\n")
    print("=" * 80)
    print("FINAL COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<22} {'Mean Reward':>16} {'Median':>10} {'Failure Rate':>14}")
    print("-" * 80)
    print(
        f"{'Fixed Timing (10s)':<22} "
        f"{results_fixed['reward'][0]:>8.2f}±{results_fixed['reward'][1]:<6.2f} "
        f"{results_fixed['median_reward']:>10.2f} "
        f"{results_fixed['failure_rate']:>12.1f}%"
    )
    print(
        f"{'MAPPO (no GAT)':<22} "
        f"{results_mlp['reward'][0]:>8.2f}±{results_mlp['reward'][1]:<6.2f} "
        f"{results_mlp['median_reward']:>10.2f} "
        f"{results_mlp['failure_rate']:>12.1f}%"
    )
    print(
        f"{'MAPPO + GAT':<22} "
        f"{results_gat['reward'][0]:>8.2f}±{results_gat['reward'][1]:<6.2f} "
        f"{results_gat['median_reward']:>10.2f} "
        f"{results_gat['failure_rate']:>12.1f}%"
    )
    print("=" * 80)