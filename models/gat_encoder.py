import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder.
    
    Takes raw node observations and graph structure,
    outputs enriched per-agent embeddings that capture
    both local state and neighborhood context.
    
    Architecture:
        Linear projection → GAT Layer 1 → GAT Layer 2 → Output embedding
    """

    def __init__(self, config):
        super(GATEncoder, self).__init__()

        self.input_dim = config['obs_size']          # 11
        self.hidden_dim = config['gat_hidden_dim']   # 64
        self.output_dim = config['embedding_dim']    # 128
        self.num_heads = config['gat_heads']         # 4
        self.num_layers = config['gat_layers']       # 2

        # --- Input projection ---
        # First we project raw observations to hidden_dim
        # This is a simple linear layer, no graph involved yet
        # Think of it as "preparing" the features before graph processing
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)

        # --- GAT Layer 1 ---
        # Input: hidden_dim (64)
        # Output: hidden_dim (64) per head × num_heads heads = 256 total
        # concat=True means we concatenate head outputs
        self.gat1 = GATConv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim // self.num_heads,  # 16 per head
            heads=self.num_heads,                             # 4 heads
            concat=True,                                      # concatenate
            dropout=0.1,
            add_self_loops=True    # each node also attends to itself
        )
        # output size = (hidden_dim // num_heads) × num_heads = hidden_dim = 64

        # --- GAT Layer 2 ---
        # Input: hidden_dim (64)
        # Output: output_dim (128)
        # concat=False means we average head outputs
        self.gat2 = GATConv(
            in_channels=self.hidden_dim,
            out_channels=self.output_dim // self.num_heads,  # 32 per head
            heads=self.num_heads,                             # 4 heads
            concat=True,                                      # concatenate → 128
            dropout=0.1,
            add_self_loops=True
        )
        # output size = (output_dim // num_heads) × num_heads = output_dim = 128

        # --- Layer normalization ---
        # Stabilizes training, very important for GNNs
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.output_dim)

        # --- Final projection ---
        # Clean linear layer after GNN to get final embedding
        self.output_projection = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, x, edge_index, return_attention=False):
        """
        Forward pass through the GAT encoder.

        Args:
            x: node features tensor, shape (num_nodes, input_dim)
               = (12, 11) in our case
            edge_index: graph connectivity in COO format, shape (2, num_edges)
                       = (2, 32) in our case
            return_attention: if True, also return attention weights
                            useful for visualization and analysis

        Returns:
            embeddings: shape (num_nodes, output_dim) = (12, 128)
            attention_weights: (optional) attention weights per edge
        """

        # --- Step 1: Input projection ---
        # (12, 11) → (12, 64)
        x = self.input_projection(x)
        x = F.elu(x)   # ELU activation, works well with GAT

        # --- Step 2: First GAT layer ---
        # Message passing round 1
        # Each node aggregates info from direct neighbors
        if return_attention:
            x_out, (edge_idx, attn_w1) = self.gat1(
                x, edge_index, return_attention_weights=True
            )
        else:
            x_out = self.gat1(x, edge_index)

        # residual connection + normalization
        # residual means we add the input back to the output
        # this helps gradients flow during training (same idea as ResNet)
        x = self.norm1(x_out + x)
        x = F.elu(x)

        # --- Step 3: Second GAT layer ---
        # Message passing round 2
        # Now each node has info from 2 hops away
        if return_attention:
            x_out, (edge_idx, attn_w2) = self.gat2(
                x, edge_index, return_attention_weights=True
            )
        else:
            x_out = self.gat2(x, edge_index)

        x = self.norm2(x_out)
        x = F.elu(x)

        # --- Step 4: Output projection ---
        # Final linear transformation
        # (12, 128) → (12, 128)
        embeddings = self.output_projection(x)

        if return_attention:
            return embeddings, (edge_idx, attn_w1, attn_w2)

        return embeddings


class GraphBuilder:
    """
    Builds PyTorch Geometric graph objects from environment data.
    
    This is a helper class that sits between the environment
    and the GNN — it takes raw observations and graph structure
    and packages them into the format PyG expects.
    """

    def __init__(self, edge_index, num_nodes, device):
        """
        Args:
            edge_index: list of (src, dst) tuples from environment
            num_nodes: total number of agents/intersections
            device: cuda or cpu
        """
        self.num_nodes = num_nodes
        self.device = device

        # convert edge list to tensor once — it never changes
        # the road network is static, only traffic changes
        src, dst = zip(*edge_index) if edge_index else ([], [])
        self.edge_index = torch.tensor(
            [list(src), list(dst)],
            dtype=torch.long,
            device=device
        )

    def build(self, observations, agent_order):
        """
        Build node feature matrix from current observations.

        Args:
            observations: dict {agent_id: obs_array}
            agent_order: list of agent IDs in consistent order

        Returns:
            x: node feature tensor, shape (num_nodes, obs_size)
            edge_index: graph connectivity tensor, shape (2, num_edges)
        """
        # stack observations in consistent agent order
        # this ensures node index i always corresponds to the same agent
        node_features = torch.stack([
            torch.tensor(observations[agent], dtype=torch.float32)
            for agent in agent_order
        ]).to(self.device)

        return node_features, self.edge_index