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
        )

        self.possible_agents = self._sumo_env.possible_agents
        self.agents = self.possible_agents[:]

        # find the maximum observation size across all agents
        # this is key — we pad everyone to the same size
        self.obs_size = max(
            self._sumo_env.observation_space(agent).shape[0]
            for agent in self.possible_agents
        )
        self._num_agents = len(self.possible_agents)

        print(f"Environment initialized with {self._num_agents} agents")
        print(f"Agents: {self.agents}")
        print(f"Unified observation size: {self.obs_size}")
        print(f"Action space sample: {self._sumo_env.action_space(self.agents[0])}")

    def _pad_observation(self, obs, agent):
        """
        Pad observation to unified size with zeros.
        This ensures all agents have the same input dimension
        which is required by our GNN.
        """
        current_size = obs.shape[0]
        if current_size < self.obs_size:
            padding = np.zeros(self.obs_size - current_size, dtype=np.float32)
            obs = np.concatenate([obs, padding])
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

        # pad all observations
        observations = {
            agent: self._pad_observation(obs, agent)
            for agent, obs in observations.items()
        }
        return observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = \
            self._sumo_env.step(actions)

        # pad all observations
        observations = {
            agent: self._pad_observation(obs, agent)
            for agent, obs in observations.items()
        }

        return observations, rewards, terminations, truncations, infos

    def close(self):
        self._sumo_env.close()

    def get_graph_structure(self):
        """
        Build the graph structure of the road network.
        Returns edge_index in COO format for PyTorch Geometric.
        
        This is the adjacency information our GNN will use.
        Each intersection is a node, each road connection is an edge.
        """
        # map agent IDs to integer indices
        agent_to_idx = {agent: i for i, agent in enumerate(self.possible_agents)}

        # define neighborhood connections based on grid structure
        # agents that are physically adjacent share an edge in our graph
        edges_src = []
        edges_dst = []

        agents = self.possible_agents

        for i, agent_i in enumerate(agents):
            for j, agent_j in enumerate(agents):
                if i != j:
                    # two intersections are neighbors if they share a direct road
                    # we determine this by checking SUMO's road network
                    # for now we connect agents that are adjacent in the grid
                    # we'll refine this using TraCI later
                    if self._are_neighbors(agent_i, agent_j):
                        edges_src.append(i)
                        edges_dst.append(j)

        return edges_src, edges_dst, agent_to_idx

    def _are_neighbors(self, agent_i, agent_j):
        """
        Two intersections are neighbors if they are directly connected by a road.
        We use naming convention from SUMO grid: A0, A1, B0, B1 etc.
        Adjacent means same row +-1 col, or same col +-1 row.
        """
        # extract row and column from agent names like 'A1', 'B2'
        try:
            row_i, col_i = agent_i[0], int(agent_i[1:])
            row_j, col_j = agent_j[0], int(agent_j[1:])

            same_row = row_i == row_j and abs(col_i - col_j) == 1
            same_col = col_i == col_j and abs(ord(row_i) - ord(row_j)) == 1

            return same_row or same_col
        except:
            return False