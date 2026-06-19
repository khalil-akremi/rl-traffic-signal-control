import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    def __init__(self, config):
        super(GATEncoder, self).__init__()

        self.input_dim = config['obs_size']          # 11
        self.hidden_dim = config['gat_hidden_dim']   # 64
        self.output_dim = config['embedding_dim']    # 128
        self.num_heads = config['gat_heads']         # 8
        self.num_layers = config['gat_layers']       # 2
        self.num_nodes = config['num_agents']        # 12
        self.id_embed_dim = config.get('id_embed_dim', 8)

        # --- Node identity embedding ---
        # tells each agent WHO it is — critical for breaking symmetry
        self.node_id_embedding = nn.Embedding(self.num_nodes, self.id_embed_dim)

        # --- Input projection ---
        # input_dim + id_embed_dim → hidden_dim
        self.input_projection = nn.Linear(
            self.input_dim + self.id_embed_dim,
            self.hidden_dim
        )

        # --- GAT Layer 1 ---
        self.gat1 = GATConv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim // self.num_heads,
            heads=self.num_heads,
            concat=True,
            dropout=0.1,
            add_self_loops=True
        )

        # --- GAT Layer 2 ---
        self.gat2 = GATConv(
            in_channels=self.hidden_dim,
            out_channels=self.output_dim // self.num_heads,
            heads=self.num_heads,
            concat=True,
            dropout=0.1,
            add_self_loops=True
        )

        # --- Layer normalization ---
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.output_dim)

        # --- Final projection ---
        self.output_projection = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, x, edge_index, node_ids=None, return_attention=False):
        """
        Args:
            x: node features, shape (num_nodes, input_dim)
            edge_index: graph connectivity, shape (2, num_edges)
            node_ids: node indices for identity embedding, shape (num_nodes,)
                      if None, inferred from x.shape[0]
            return_attention: if True, return attention weights
        """

        # --- Step 1: Inject node identity ---
        if node_ids is None:
            node_ids = torch.arange(x.shape[0], device=x.device)
        id_emb = self.node_id_embedding(node_ids)   # (num_nodes, id_embed_dim)
        x = torch.cat([x, id_emb], dim=-1)          # (num_nodes, input_dim + id_embed_dim)

        # --- Step 2: Input projection ---
        x = self.input_projection(x)                 # (num_nodes, hidden_dim)
        x = F.elu(x)

        # --- Step 3: First GAT layer ---
        if return_attention:
            x_out, (edge_idx, attn_w1) = self.gat1(
                x, edge_index, return_attention_weights=True
            )
        else:
            x_out = self.gat1(x, edge_index)

        x = self.norm1(x_out + x)
        x = F.elu(x)

        # --- Step 4: Second GAT layer ---
        if return_attention:
            x_out, (edge_idx, attn_w2) = self.gat2(
                x, edge_index, return_attention_weights=True
            )
        else:
            x_out = self.gat2(x, edge_index)

        x = self.norm2(x_out)
        x = F.elu(x)

        # --- Step 5: Output projection ---
        embeddings = self.output_projection(x)       # (num_nodes, output_dim)

        if return_attention:
            return embeddings, (edge_idx, attn_w1, attn_w2)

        return embeddings


class GraphBuilder:
    def __init__(self, edge_index, num_nodes, device):
        self.num_nodes = num_nodes
        self.device = device

        src, dst = zip(*edge_index) if edge_index else ([], [])
        self.edge_index = torch.tensor(
            [list(src), list(dst)],
            dtype=torch.long,
            device=device
        )

        # node ids — precomputed once, never changes
        self.node_ids = torch.arange(num_nodes, dtype=torch.long, device=device)

    def build(self, observations, agent_order):
        """
        Build node feature matrix from current observations.

        Returns:
            x: node features, shape (num_nodes, obs_size)
            edge_index: graph connectivity, shape (2, num_edges)
            node_ids: node indices, shape (num_nodes,)
        """
        node_features = torch.stack([
            torch.tensor(observations[agent], dtype=torch.float32)
            for agent in agent_order
        ]).to(self.device)

        return node_features, self.edge_index, self.node_ids

class MLPEncoder(nn.Module):
    """
    Baseline encoder with NO graph structure.
    Each agent's embedding depends only on its own observation
    (plus its identity embedding for fairness with the GAT version).
    Used to measure how much the GAT's neighbor-aggregation
    actually contributes versus pure independent local learning.
    """

    def __init__(self, config):
        super(MLPEncoder, self).__init__()

        self.input_dim = config['obs_size']
        self.hidden_dim = config['gat_hidden_dim']
        self.output_dim = config['embedding_dim']
        self.num_nodes = config['num_agents']
        self.id_embed_dim = config.get('id_embed_dim', 8)

        self.node_id_embedding = nn.Embedding(self.num_nodes, self.id_embed_dim)

        self.network = nn.Sequential(
            nn.Linear(self.input_dim + self.id_embed_dim, self.hidden_dim),
            nn.ELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ELU(),
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.output_dim)
        )

    def forward(self, x, edge_index=None, node_ids=None, return_attention=False):
        # edge_index is intentionally ignored — no neighbor information used
        if node_ids is None:
            node_ids = torch.arange(x.shape[0], device=x.device)
        id_emb = self.node_id_embedding(node_ids)
        x = torch.cat([x, id_emb], dim=-1)
        embeddings = self.network(x)
        return embeddings