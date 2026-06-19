import sumo_rl
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces


class TrafficEnvironment(ParallelEnv):
    """
    Multi-agent traffic environment wrapping SUMO-RL.
    Each agent controls one intersection in a 4x4 grid.
    Follows PettingZoo Parallel API.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self._sumo_env = sumo_rl.parallel_env(
            net_file=config['net_file'],
            route_file=config['route_file'],
            use_gui=config.get('use_gui', False),
            num_seconds=config.get('num_seconds', 3600),
            delta_time=config.get('delta_time', 5),
            yellow_time=config.get('yellow_time', 2),
            min_green=config.get('min_green', 5),
            reward_fn=self._compute_reward,
        )

        self.possible_agents = self._sumo_env.possible_agents
        self.agents = self.possible_agents[:]

        self.obs_size = max(
            self._sumo_env.observation_space(agent).shape[0]
            for agent in self.possible_agents
        )
        self._num_agents = len(self.possible_agents)

        self.training_mode = True

        print(f"Environment initialized with {self._num_agents} agents")
        print(f"Agents: {self.agents}")
        print(f"Unified observation size: {self.obs_size}")
        print(f"Action space sample: {self._sumo_env.action_space(self.agents[0])}")

    def _compute_reward(self, traffic_signal):
        """
        Pressure-based reward.
        Counts halting vehicles on all incoming lanes.
        More sensitive than raw waiting time — stronger learning signal.
        Negative because we want to minimize pressure.
        """
        total_pressure = 0
        for lane in traffic_signal.lanes:
            halting = traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)
            total_pressure += halting
        return -total_pressure / 10.0

    def _pad_observation(self, obs, agent, training=True):
        """
        Pad observation to unified size with zeros.
        Optionally add small noise during training to prevent
        the policy from locking onto deterministic feature shortcuts.
        """
        current_size = obs.shape[0]
        if current_size < self.obs_size:
            padding = np.zeros(self.obs_size - current_size, dtype=np.float32)
            obs = np.concatenate([obs, padding])

        if training and self.config.get('obs_noise', 0.0) > 0:
            noise = np.random.normal(
                0,
                self.config['obs_noise'],
                size=obs.shape
            ).astype(np.float32)
            obs = np.clip(obs + noise, 0.0, 1.0)

        return obs

    def observation_space(self, agent):
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_size,),
            dtype=np.float32
        )

    def action_space(self, agent):
        return self._sumo_env.action_space(agent)

    def reset(self, seed=None, options=None):
        observations, infos = self._sumo_env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]

        # stagger phases to break symmetry
        inner = None
        try:
            inner = self._sumo_env.aec_env.env.env.env

            for i, agent in enumerate(self.possible_agents):
                ts = inner.traffic_signals[agent]
                target_phase = i % ts.num_green_phases

                ts.green_phase = target_phase
                ts.time_since_last_phase_change = 0
                ts.sumo.trafficlight.setRedYellowGreenState(
                    agent, ts.all_phases[target_phase].state
                )

            print(f"Phase staggering successful for {len(self.possible_agents)} agents")
        except Exception as e:
            print(f"WARNING: Phase staggering failed: {e}")

        # re-fetch observations AFTER staggering, since obs depend on green_phase
        if inner is not None:
            observations = {
                agent: self._pad_observation(
                    inner.traffic_signals[agent].compute_observation(), agent,
                    training=self.training_mode
                )
                for agent in self.possible_agents
            }
        else:
            observations = {
                agent: self._pad_observation(obs, agent, training=self.training_mode)
                for agent, obs in observations.items()
            }

        return observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = \
            self._sumo_env.step(actions)
        observations = {
            agent: self._pad_observation(obs, agent, training=self.training_mode)
            for agent, obs in observations.items()
        }
        return observations, rewards, terminations, truncations, infos

    def close(self):
        self._sumo_env.close()

    def set_train_mode(self):
        self.training_mode = True

    def set_eval_mode(self):
        self.training_mode = False

    def get_graph_structure(self):
        """
        Build the graph structure of the road network.
        Returns edge_index in COO format for PyTorch Geometric.
        Each intersection is a node, each road connection is an edge.
        """
        agent_to_idx = {agent: i for i, agent in enumerate(self.possible_agents)}

        edges_src = []
        edges_dst = []

        agents = self.possible_agents
        for i, agent_i in enumerate(agents):
            for j, agent_j in enumerate(agents):
                if i != j:
                    if self._are_neighbors(agent_i, agent_j):
                        edges_src.append(i)
                        edges_dst.append(j)

        return edges_src, edges_dst, agent_to_idx

    def _are_neighbors(self, agent_i, agent_j):
        """
        Two intersections are neighbors if directly connected by a road.
        Uses naming convention: A1, B2, C3 etc.
        Adjacent = same row +-1 col, or same col +-1 row.
        """
        try:
            row_i, col_i = agent_i[0], int(agent_i[1:])
            row_j, col_j = agent_j[0], int(agent_j[1:])

            same_row = row_i == row_j and abs(col_i - col_j) == 1
            same_col = col_i == col_j and abs(ord(row_i) - ord(row_j)) == 1

            return same_row or same_col
        except:
            return False