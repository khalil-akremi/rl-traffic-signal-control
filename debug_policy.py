import torch
import yaml
import numpy as np
from environment.traffic_env import TrafficEnvironment
from models.gat_encoder import GATEncoder, GraphBuilder
from models.mappo import MAPPO


def debug_policy(config_path='configs/config.yaml'):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = TrafficEnvironment(config['environment'])
    edges_src, edges_dst, _ = env.get_graph_structure()
    edge_pairs = list(zip(edges_src, edges_dst))
    graph_builder = GraphBuilder(edge_pairs, env._num_agents, device)

    gnn_config = {'obs_size': env.obs_size, **config['model']}
    encoder = GATEncoder(gnn_config).to(device)

    mappo_config = {
        **config['mappo'],
        'embedding_dim': config['model']['embedding_dim'],
        'action_dim': 2
    }
    mappo = MAPPO(mappo_config, env._num_agents, device)

    checkpoint = torch.load(
        'checkpoints/best_model.pt',
        map_location=device,
        weights_only=False
    )
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    mappo.actor.load_state_dict(checkpoint['actor_state_dict'])
    encoder.eval()
    mappo.actor.eval()

    observations, _ = env.reset(seed=42)

    # advance 200 steps so traffic builds up
    print("Advancing simulation 200 steps...")
    for step in range(200):
        actions = {agent: env.action_space(agent).sample()
                   for agent in env.possible_agents}
        observations, _, terminations, truncations, _ = env.step(actions)
        if all(terminations.values()) or all(truncations.values()):
            break

    print(f"Simulation at step 200")

    # now build graph from real traffic state
    node_features, edge_index, node_ids = graph_builder.build(
        observations, env.possible_agents
    )

    # test 1 — are observations different now?
    obs_std = node_features.std(dim=0).mean().item()
    print(f"\n--- TEST 1: Observation variance after 200 steps ---")
    print(f"Obs std across agents: {obs_std:.6f}")
    for i, agent in enumerate(env.possible_agents):
        print(f"  {agent}: {node_features[i].cpu().numpy().round(3)}")

    # test 2 — are embeddings different?
    with torch.no_grad():
        embeddings = encoder(node_features, edge_index, node_ids)

    emb_std = embeddings.std(dim=0).mean().item()
    print(f"\n--- TEST 2: Embedding variance after 200 steps ---")
    print(f"Embedding std across agents: {emb_std:.6f}")
    print("First 5 values per agent:")
    for i, agent in enumerate(env.possible_agents):
        print(f"  {agent}: {embeddings[i][:5].cpu().numpy().round(4)}")

    # test 3 — are logits different?
    with torch.no_grad():
        dist, logits = mappo.actor(embeddings.float())

    print(f"\n--- TEST 3: Logits and probabilities after 200 steps ---")
    print("Logits per agent:")
    for i, agent in enumerate(env.possible_agents):
        print(f"  {agent}: {logits[i].cpu().numpy().round(4)}")

    print("\nAction probabilities per agent:")
    for i, agent in enumerate(env.possible_agents):
        print(f"  {agent}: {dist.probs[i].cpu().numpy().round(4)}")

    # summary
    logit_std = logits.std(dim=0).mean().item()
    print(f"\n--- SUMMARY ---")
    print(f"Obs std:      {obs_std:.6f}")
    print(f"Embedding std:{emb_std:.6f}")
    print(f"Logit std:    {logit_std:.6f}")

    if obs_std > 0.01 and emb_std < 1e-4:
        print("\nDIAGNOSIS: GAT collapsing all agents to same representation")
    elif obs_std > 0.01 and emb_std > 1e-4 and logit_std < 1e-4:
        print("\nDIAGNOSIS: Actor collapsed — ignoring embeddings")
    elif obs_std > 0.01 and emb_std > 1e-4 and logit_std > 1e-4:
        print("\nDIAGNOSIS: Policy is NOT collapsed — working correctly")
    else:
        print("\nDIAGNOSIS: Observations still too uniform — need more steps")

    env.close()
    
def count_actions_during_eval(config_path='configs/config.yaml', steps=500):
    """
    Run the trained model for 500 steps and count how often
    each action is selected. Compare against fixed timing.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = TrafficEnvironment(config['environment'])
    edges_src, edges_dst, _ = env.get_graph_structure()
    edge_pairs = list(zip(edges_src, edges_dst))
    graph_builder = GraphBuilder(edge_pairs, env._num_agents, device)

    gnn_config = {'obs_size': env.obs_size, **config['model']}
    encoder = GATEncoder(gnn_config).to(device)
    mappo_config = {
        **config['mappo'],
        'embedding_dim': config['model']['embedding_dim'],
        'action_dim': 2
    }
    mappo = MAPPO(mappo_config, env._num_agents, device)

    checkpoint = torch.load(
        'checkpoints/best_model.pt',
        map_location=device,
        weights_only=False
    )
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    mappo.actor.load_state_dict(checkpoint['actor_state_dict'])
    encoder.eval()
    mappo.actor.eval()

    observations, _ = env.reset(seed=42)

    # fixed timing reference
    delta_time = config['environment']['delta_time']
    steps_per_phase = 10 // delta_time  # 10s fixed timing

    # counters
    model_action_counts = {agent: {0: 0, 1: 0}
                           for agent in env.possible_agents}
    fixed_action_counts = {0: 0, 1: 0}
    agreements = 0
    total = 0

    print(f"Running {steps} steps and counting actions...")

    for step in range(steps):
        node_features, edge_index, node_ids = graph_builder.build(
            observations, env.possible_agents
        )

        with torch.no_grad():
            embeddings = encoder(node_features, edge_index, node_ids)
            dist, _ = mappo.actor(embeddings.float())
            actions_tensor = dist.probs.argmax(dim=-1)

        model_actions = {
            agent: actions_tensor[i].item()
            for i, agent in enumerate(env.possible_agents)
        }

        # fixed timing action at this step
        fixed_action = (step // steps_per_phase) % 2

        # count agreements
        for agent in env.possible_agents:
            model_action_counts[agent][model_actions[agent]] += 1
            if model_actions[agent] == fixed_action:
                agreements += 1
            total += 1

        fixed_action_counts[fixed_action] += 1

        observations, _, terminations, truncations, _ = \
            env.step(model_actions)

        if all(terminations.values()) or all(truncations.values()):
            break

    # results
    print(f"\n--- ACTION DISTRIBUTION (model) ---")
    total_0 = sum(v[0] for v in model_action_counts.values())
    total_1 = sum(v[1] for v in model_action_counts.values())
    grand_total = total_0 + total_1
    print(f"Action 0: {total_0} ({100*total_0/grand_total:.1f}%)")
    print(f"Action 1: {total_1} ({100*total_1/grand_total:.1f}%)")

    print(f"\n--- ACTION DISTRIBUTION (fixed 10s timing) ---")
    ft = fixed_action_counts[0] + fixed_action_counts[1]
    print(f"Action 0: {fixed_action_counts[0]} ({100*fixed_action_counts[0]/ft:.1f}%)")
    print(f"Action 1: {fixed_action_counts[1]} ({100*fixed_action_counts[1]/ft:.1f}%)")

    print(f"\n--- AGREEMENT RATE ---")
    print(f"Model agrees with fixed timing: {100*agreements/total:.1f}% of steps")

    print(f"\n--- PER AGENT ACTION COUNTS ---")
    for agent in env.possible_agents:
        a0 = model_action_counts[agent][0]
        a1 = model_action_counts[agent][1]
        total_agent = a0 + a1
        print(f"  {agent}: action0={a0}({100*a0/total_agent:.1f}%) "
              f"action1={a1}({100*a1/total_agent:.1f}%)")

    env.close()
def check_action_synchrony(config_path='configs/config.yaml', steps=100):
    """
    Check if agents switch actions at the same timesteps.
    If yes → synchronized behavior despite different embeddings.
    If no → truly independent decisions.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = TrafficEnvironment(config['environment'])
    edges_src, edges_dst, _ = env.get_graph_structure()
    edge_pairs = list(zip(edges_src, edges_dst))
    graph_builder = GraphBuilder(edge_pairs, env._num_agents, device)

    gnn_config = {'obs_size': env.obs_size, **config['model']}
    encoder = GATEncoder(gnn_config).to(device)
    mappo_config = {
        **config['mappo'],
        'embedding_dim': config['model']['embedding_dim'],
        'action_dim': 2
    }
    mappo = MAPPO(mappo_config, env._num_agents, device)

    checkpoint = torch.load(
        'checkpoints/best_model.pt',
        map_location=device,
        weights_only=False
    )
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    mappo.actor.load_state_dict(checkpoint['actor_state_dict'])
    encoder.eval()
    mappo.actor.eval()

    observations, _ = env.reset(seed=42)

    # record actions per agent per step
    action_log = {agent: [] for agent in env.possible_agents}

    print(f"Recording actions for {steps} steps...")
    for step in range(steps):
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

        for agent in env.possible_agents:
            action_log[agent].append(actions_dict[agent])

        observations, _, terminations, truncations, _ = \
            env.step(actions_dict)
        if all(terminations.values()) or all(truncations.values()):
            break

    # check synchrony — do all agents switch at the same step?
    print(f"\n--- ACTION SEQUENCE (first 30 steps) ---")
    print(f"{'Step':<6}", end="")
    for agent in env.possible_agents:
        print(f"{agent:>5}", end="")
    print()

    for step in range(min(30, steps)):
        print(f"{step:<6}", end="")
        for agent in env.possible_agents:
            print(f"{action_log[agent][step]:>5}", end="")
        print()

    # count how many steps all agents agree
    all_same = sum(
        1 for step in range(len(action_log[env.possible_agents[0]]))
        if len(set(action_log[agent][step]
                   for agent in env.possible_agents)) == 1
    )
    total = len(action_log[env.possible_agents[0]])

    print(f"\n--- SYNCHRONY ANALYSIS ---")
    print(f"Steps where ALL agents chose same action: "
          f"{all_same}/{total} ({100*all_same/total:.1f}%)")
    print(f"Steps where agents DISAGREED: "
          f"{total-all_same}/{total} ({100*(total-all_same)/total:.1f}%)")

    env.close()
def test_stochastic_vs_greedy(config_path='configs/config.yaml', steps=30):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = TrafficEnvironment(config['environment'])
    edges_src, edges_dst, _ = env.get_graph_structure()
    edge_pairs = list(zip(edges_src, edges_dst))
    graph_builder = GraphBuilder(edge_pairs, env._num_agents, device)

    gnn_config = {'obs_size': env.obs_size, **config['model']}
    encoder = GATEncoder(gnn_config).to(device)
    mappo_config = {
        **config['mappo'],
        'embedding_dim': config['model']['embedding_dim'],
        'action_dim': 2
    }
    mappo = MAPPO(mappo_config, env._num_agents, device)

    checkpoint = torch.load(
        'checkpoints/best_model.pt',
        map_location=device,
        weights_only=False
    )
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    mappo.actor.load_state_dict(checkpoint['actor_state_dict'])
    encoder.eval()
    mappo.actor.eval()

    observations, _ = env.reset(seed=42)

    print(f"\n{'Step':<6} {'GREEDY (first 6 agents)':^36} {'STOCHASTIC (first 6 agents)':^36}")
    print(f"{'':6} {' '.join(env.possible_agents[:6]):^36} {' '.join(env.possible_agents[:6]):^36}")

    for step in range(steps):
        node_features, edge_index, node_ids = graph_builder.build(
            observations, env.possible_agents
        )
        with torch.no_grad():
            embeddings = encoder(node_features, edge_index, node_ids)
            dist, _ = mappo.actor(embeddings.float())
            greedy_actions = dist.probs.argmax(dim=-1)
            stochastic_actions = dist.sample()

        greedy_str = ' '.join(
            str(greedy_actions[i].item())
            for i in range(min(6, len(env.possible_agents)))
        )
        stochastic_str = ' '.join(
            str(stochastic_actions[i].item())
            for i in range(min(6, len(env.possible_agents)))
        )

        print(f"{step:<6} {greedy_str:^36} {stochastic_str:^36}")

        actions_dict = {
            agent: greedy_actions[i].item()
            for i, agent in enumerate(env.possible_agents)
        }
        observations, _, terminations, truncations, _ = env.step(actions_dict)
        if all(terminations.values()) or all(truncations.values()):
            break

    env.close()
def systematic_diagnosis(config_path='configs/config.yaml', steps=100):
    """
    Four systematic checks before concluding root cause of synchrony.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = TrafficEnvironment(config['environment'])
    edges_src, edges_dst, _ = env.get_graph_structure()
    edge_pairs = list(zip(edges_src, edges_dst))
    graph_builder = GraphBuilder(edge_pairs, env._num_agents, device)

    gnn_config = {'obs_size': env.obs_size, **config['model']}
    encoder = GATEncoder(gnn_config).to(device)
    mappo_config = {
        **config['mappo'],
        'embedding_dim': config['model']['embedding_dim'],
        'action_dim': 2
    }
    mappo = MAPPO(mappo_config, env._num_agents, device)

    checkpoint = torch.load(
        'checkpoints/best_model.pt',
        map_location=device,
        weights_only=False
    )
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    mappo.actor.load_state_dict(checkpoint['actor_state_dict'])
    encoder.eval()
    mappo.actor.eval()

    # =============================================
    # CHECK 1 — Entropy per agent over time
    # =============================================
    print("\n" + "="*60)
    print("CHECK 1: Policy entropy per agent")
    print("="*60)

    observations, _ = env.reset(seed=42)
    entropies_per_agent = {agent: [] for agent in env.possible_agents}

    for step in range(steps):
        node_features, edge_index, node_ids = graph_builder.build(
            observations, env.possible_agents
        )
        with torch.no_grad():
            embeddings = encoder(node_features, edge_index, node_ids)
            dist, _ = mappo.actor(embeddings.float())
            entropy_per_agent = dist.entropy()

        for i, agent in enumerate(env.possible_agents):
            entropies_per_agent[agent].append(
                entropy_per_agent[i].item()
            )

        actions_dict = {
            agent: dist.probs.argmax(dim=-1)[i].item()
            for i, agent in enumerate(env.possible_agents)
        }
        observations, _, terminations, truncations, _ = env.step(actions_dict)
        if all(terminations.values()) or all(truncations.values()):
            break

    print(f"{'Agent':<6} {'Mean Entropy':>14} {'Min':>8} {'Max':>8}")
    print("-" * 40)
    for agent in env.possible_agents:
        e = entropies_per_agent[agent]
        print(f"{agent:<6} {np.mean(e):>14.6f} {np.min(e):>8.6f} {np.max(e):>8.6f}")

    overall_entropy = np.mean([
        np.mean(v) for v in entropies_per_agent.values()
    ])
    print(f"\nOverall mean entropy: {overall_entropy:.6f}")
    print(f"Max possible entropy (2 actions): {np.log(2):.6f}")
    print(f"Entropy ratio: {overall_entropy/np.log(2)*100:.1f}%")

    # =============================================
    # CHECK 2 — Feature ablation: remove current_phase
    # =============================================
    print("\n" + "="*60)
    print("CHECK 2: Feature ablation — zero out current_phase")
    print("="*60)
    print("(Feature index 0 = current phase in SUMO-RL observations)")

    observations, _ = env.reset(seed=42)
    sync_normal = 0
    sync_ablated = 0
    total = 0

    for step in range(steps):
        node_features, edge_index, node_ids = graph_builder.build(
            observations, env.possible_agents
        )

        # normal forward pass
        with torch.no_grad():
            embeddings_normal = encoder(node_features, edge_index, node_ids)
            dist_normal, _ = mappo.actor(embeddings_normal.float())
            actions_normal = dist_normal.probs.argmax(dim=-1)

        # ablated — zero out first feature (current phase)
        node_features_ablated = node_features.clone()
        node_features_ablated[:, 0] = 0.0

        with torch.no_grad():
            embeddings_ablated = encoder(node_features_ablated, edge_index, node_ids)
            dist_ablated, _ = mappo.actor(embeddings_ablated.float())
            actions_ablated = dist_ablated.probs.argmax(dim=-1)

        # check synchrony
        normal_unique = len(set(actions_normal.tolist()))
        ablated_unique = len(set(actions_ablated.tolist()))

        if normal_unique == 1:
            sync_normal += 1
        if ablated_unique == 1:
            sync_ablated += 1
        total += 1

        actions_dict = {
            agent: actions_normal[i].item()
            for i, agent in enumerate(env.possible_agents)
        }
        observations, _, terminations, truncations, _ = env.step(actions_dict)
        if all(terminations.values()) or all(truncations.values()):
            break

    print(f"Normal policy synchronized: {sync_normal}/{total} "
          f"({100*sync_normal/total:.1f}%)")
    print(f"Ablated policy synchronized: {sync_ablated}/{total} "
          f"({100*sync_ablated/total:.1f}%)")
    if sync_ablated < sync_normal:
        print("→ current_phase DOES contribute to synchrony")
    elif sync_ablated == sync_normal:
        print("→ current_phase does NOT cause synchrony")
    else:
        print("→ removing current_phase made synchrony WORSE")

    # =============================================
    # CHECK 3 — Per-agent perturbation test
    # =============================================
    print("\n" + "="*60)
    print("CHECK 3: Single agent perturbation")
    print("="*60)
    print("Adding noise to A1 only — do other agents deviate?")

    observations, _ = env.reset(seed=42)
    deviations = []

    for step in range(steps):
        node_features, edge_index, node_ids = graph_builder.build(
            observations, env.possible_agents
        )

        # normal actions
        with torch.no_grad():
            embeddings = encoder(node_features, edge_index, node_ids)
            dist, _ = mappo.actor(embeddings.float())
            actions_normal = dist.probs.argmax(dim=-1)

        # perturb only agent 0 (A1)
        node_features_perturbed = node_features.clone()
        node_features_perturbed[0] += torch.randn_like(
            node_features_perturbed[0]
        ) * 0.5  # strong noise

        with torch.no_grad():
            embeddings_perturbed = encoder(
                node_features_perturbed, edge_index, node_ids
            )
            dist_perturbed, _ = mappo.actor(embeddings_perturbed.float())
            actions_perturbed = dist_perturbed.probs.argmax(dim=-1)

        # how many agents changed action?
        changed = (actions_normal != actions_perturbed).sum().item()
        deviations.append(changed)

        actions_dict = {
            agent: actions_normal[i].item()
            for i, agent in enumerate(env.possible_agents)
        }
        observations, _, terminations, truncations, _ = env.step(actions_dict)
        if all(terminations.values()) or all(truncations.values()):
            break

    avg_changed = np.mean(deviations)
    print(f"Average agents changing action when A1 is perturbed: "
          f"{avg_changed:.2f} / {env._num_agents}")
    print(f"Steps where ONLY A1 changed: "
          f"{sum(1 for d in deviations if d == 1)}/{len(deviations)}")
    print(f"Steps where ALL agents changed: "
          f"{sum(1 for d in deviations if d == env._num_agents)}/{len(deviations)}")

    if avg_changed <= 1.5:
        print("→ Perturbation is LOCAL — agents react independently")
    elif avg_changed >= env._num_agents * 0.8:
        print("→ Perturbation is GLOBAL — environment/GNN coupling is strong")
    else:
        print("→ Perturbation has MIXED effect")

    # =============================================
    # CHECK 4 — Correlation: which features drive action flips
    # =============================================
    print("\n" + "="*60)
    print("CHECK 4: Feature correlation with action flips")
    print("="*60)

    observations, _ = env.reset(seed=42)
    feature_values = []
    action_values = []

    for step in range(steps):
        node_features, edge_index, node_ids = graph_builder.build(
            observations, env.possible_agents
        )
        with torch.no_grad():
            embeddings = encoder(node_features, edge_index, node_ids)
            dist, _ = mappo.actor(embeddings.float())
            actions = dist.probs.argmax(dim=-1)

        # use first agent as reference
        feature_values.append(node_features[0].cpu().numpy())
        action_values.append(actions[0].item())

        actions_dict = {
            agent: actions[i].item()
            for i, agent in enumerate(env.possible_agents)
        }
        observations, _, terminations, truncations, _ = env.step(actions_dict)
        if all(terminations.values()) or all(truncations.values()):
            break

    feature_values = np.array(feature_values)
    action_values = np.array(action_values)

    print(f"Correlation between each feature and action (agent A1):")
    print(f"{'Feature':>8} {'Correlation':>12}")
    print("-" * 22)
    for feat_idx in range(feature_values.shape[1]):
        corr = np.corrcoef(
            feature_values[:, feat_idx],
            action_values
        )[0, 1]
        if not np.isnan(corr):
            print(f"{feat_idx:>8} {corr:>12.4f}")

    env.close()
def test_phase_randomization(config_path='configs/config.yaml', steps=50):
    """
    Randomize initial traffic light phases per intersection.
    If synchrony breaks → phase correlation was the cause.
    If synchrony persists → entropy collapse is the cause.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = TrafficEnvironment(config['environment'])
    edges_src, edges_dst, _ = env.get_graph_structure()
    edge_pairs = list(zip(edges_src, edges_dst))
    graph_builder = GraphBuilder(edge_pairs, env._num_agents, device)

    gnn_config = {'obs_size': env.obs_size, **config['model']}
    encoder = GATEncoder(gnn_config).to(device)
    mappo_config = {
        **config['mappo'],
        'embedding_dim': config['model']['embedding_dim'],
        'action_dim': 2
    }
    mappo = MAPPO(mappo_config, env._num_agents, device)

    checkpoint = torch.load(
        'checkpoints/best_model.pt',
        map_location=device,
        weights_only=False
    )
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    mappo.actor.load_state_dict(checkpoint['actor_state_dict'])
    encoder.eval()
    mappo.actor.eval()

    # reset environment
    observations, _ = env.reset(seed=42)

    # randomize initial phases per intersection
    print("Randomizing initial phases per intersection...")
    try:
        sumo = env._sumo_env.env.sumo
        import random
        random.seed(42)
        for agent in env.possible_agents:
            random_phase = random.randint(0, 1)
            sumo.trafficlight.setPhase(agent, random_phase)
            print(f"  {agent} → phase {random_phase}")
    except Exception as e:
        print(f"Phase randomization failed: {e}")
        env.close()
        return

    # now get fresh observations after phase randomization
    # step once with dummy actions to get updated observations
    dummy_actions = {agent: 0 for agent in env.possible_agents}
    observations, _, _, _, _ = env.step(dummy_actions)

    print(f"\n--- OBSERVATIONS AFTER PHASE RANDOMIZATION ---")
    node_features, edge_index, node_ids = graph_builder.build(
        observations, env.possible_agents
    )
    obs_std = node_features.std(dim=0).mean().item()
    print(f"Obs std: {obs_std:.6f}")
    for i, agent in enumerate(env.possible_agents):
        print(f"  {agent}: {node_features[i].cpu().numpy().round(3)}")

    print(f"\n--- ACTION SEQUENCE AFTER PHASE RANDOMIZATION ---")
    print(f"{'Step':<6}", end="")
    for agent in env.possible_agents:
        print(f"{agent:>5}", end="")
    print()

    sync_count = 0
    total = 0

    for step in range(steps):
        node_features, edge_index, node_ids = graph_builder.build(
            observations, env.possible_agents
        )
        with torch.no_grad():
            embeddings = encoder(node_features, edge_index, node_ids)
            dist, _ = mappo.actor(embeddings.float())
            actions_tensor = dist.probs.argmax(dim=-1)

        # print first 20 steps
        if step < 20:
            print(f"{step:<6}", end="")
            for i in range(len(env.possible_agents)):
                print(f"{actions_tensor[i].item():>5}", end="")
            print()

        # check synchrony
        unique_actions = len(set(actions_tensor.tolist()))
        if unique_actions == 1:
            sync_count += 1
        total += 1

        actions_dict = {
            agent: actions_tensor[i].item()
            for i, agent in enumerate(env.possible_agents)
        }
        observations, _, terminations, truncations, _ = env.step(actions_dict)
        if all(terminations.values()) or all(truncations.values()):
            break

    print(f"\n--- SYNCHRONY AFTER PHASE RANDOMIZATION ---")
    print(f"Synchronized steps: {sync_count}/{total} "
          f"({100*sync_count/total:.1f}%)")

    if sync_count == total:
        print("\nCONCLUSION: Synchrony persists after phase randomization")
        print("→ ENTROPY COLLAPSE is the root cause")
        print("→ Fix: increase entropy_coef and retrain")
    elif sync_count < total * 0.5:
        print("\nCONCLUSION: Synchrony mostly breaks after phase randomization")
        print("→ PHASE CORRELATION was the primary driver")
        print("→ Fix: stagger phases in environment design")
    else:
        print("\nCONCLUSION: Mixed result — both factors contribute")

    env.close()


if __name__ == '__main__':
    test_phase_randomization()


if __name__ == '__main__':
    systematic_diagnosis()


if __name__ == '__main__':
    test_stochastic_vs_greedy()


if __name__ == '__main__':
    check_action_synchrony()


if __name__ == '__main__':
    count_actions_during_eval()


if __name__ == '__main__':
    debug_policy()