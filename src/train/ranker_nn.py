import torch
import torch.nn as nn

class DLRMRanker(nn.Module):
    def __init__(self, num_dense_features, output_dims, bottom_mlp_dims=[64, 64], top_mlp_dims=[64, 32], dropout=0.2):
        super().__init__()
        
        # EXPERT_NOTE: 1. The Dense Path (Bottom MLP)
        # We transform raw numerical features into a "Latent Vector" (e.g., 64D)
        # This creates a "Level Playing Field" so scalars can interact effectively with embeddings.
        bottom_layers = []
        in_dim = num_dense_features
        for out_dim in bottom_mlp_dims:
            bottom_layers.append(nn.Linear(in_dim, out_dim))
            bottom_layers.append(nn.ReLU())
            in_dim = out_dim
        self.bottom_mlp = nn.Sequential(*bottom_layers)
        
        # The standardized dimension for all embeddings and Bottom MLP output
        self.embedding_dim = bottom_mlp_dims[-1]
        
        # EXPERT_NOTE: 2. The Sparse Path (Hashing & Embeddings)
        # Instead of `num_users` which causes OOM (Out Of Memory) at 1B users, 
        # we strictly cap the Embedding Table using a Hash Bucket.
        self.user_hash_bucket_size = 100_000 # In production, BO is used to tune this boundary
        self.post_hash_bucket_size = 100_000
        
        self.user_emb = nn.Embedding(self.user_hash_bucket_size, self.embedding_dim)
        self.post_emb = nn.Embedding(self.post_hash_bucket_size, self.embedding_dim)
        
        # Categorical features with known small cardinality don't need hashes
        self.post_type_emb = nn.Embedding(3, self.embedding_dim) # text, image, video
        self.post_hour_emb = nn.Embedding(24, self.embedding_dim) # 24 hours
        
        # EXPERT_NOTE: 3. The Interaction Layer Pre-Calculation
        # Number of vectors = 1 (Bottom MLP) + 4 (Sparse) + 2 (Two Tower Input) = 7 vectors.
        # Pairwise dot products = 7 * (7 - 1) / 2 = 21 interaction scores.
        num_interaction_scores = 21
        
        # EXPERT_NOTE: 4. The Top MLP
        # It takes the Interaction Scores AND the Bottom MLP output.
        top_in_dim = num_interaction_scores + self.embedding_dim
        top_layers = []
        for out_dim in top_mlp_dims:
            top_layers.append(nn.Linear(top_in_dim, out_dim))
            top_layers.append(nn.ReLU())
            top_layers.append(nn.Dropout(dropout))
            top_in_dim = out_dim
        top_layers.append(nn.Linear(top_in_dim, output_dims))
        self.top_mlp = nn.Sequential(*top_layers)

    def forward(self, dense_x, sparse_x, tower_x):
        """
        dense_x: (batch, num_dense_features)
        sparse_x: (batch, 4) -> [user_id, post_id, post_type, post_hour]
        tower_x: (batch, 2, emb_dim) -> [user_tower_emb, post_tower_emb]
        """
        
        # 1. Process Dense
        bottom_out = self.bottom_mlp(dense_x) # (batch, 64)
        
        # 2. Process Sparse (using Modulo Hashing trick for IDs)
        user_ids_hashed = sparse_x[:, 0] % self.user_hash_bucket_size
        post_ids_hashed = sparse_x[:, 1] % self.post_hash_bucket_size
        
        u_emb = self.user_emb(user_ids_hashed) # (batch, 64)
        p_emb = self.post_emb(post_ids_hashed) # (batch, 64)
        type_emb = self.post_type_emb(sparse_x[:, 2]) # (batch, 64)
        hour_emb = self.post_hour_emb(sparse_x[:, 3]) # (batch, 64)
        
        # 3. Process Tower (Already 64D from Two-Tower model Stage 1)
        tt_u_emb = tower_x[:, 0, :]
        tt_p_emb = tower_x[:, 1, :]
        
        # 4. Interaction Layer (Dot-Product)
        # Stack all vectors: (batch, num_vectors, 64)
        all_vectors = torch.stack([bottom_out, u_emb, p_emb, type_emb, hour_emb, tt_u_emb, tt_p_emb], dim=1)
        
        # Explicit pair-wise dot product using Batch Matrix Multiplication (bmm)
        # all_vectors: (batch, 7, 64), transpose: (batch, 64, 7)
        # interactions result: (batch, 7, 7)
        interactions = torch.bmm(all_vectors, all_vectors.transpose(1, 2))
        
        # Extract upper triangle (ignore self-interactions like User-User, and ignore duplicates)
        # Creating indices for the upper triangle
        triu_indices = torch.triu_indices(row=7, col=7, offset=1)
        interaction_scores = interactions[:, triu_indices[0], triu_indices[1]] # (batch, 21)
        
        # 5. Top MLP Combination
        # Re-attach the continuous bottom MLP signal with the pure correlation scores
        top_mlp_in = torch.cat([bottom_out, interaction_scores], dim=1) # (batch, 64 + 21)
        
        # Final prediction
        output = self.top_mlp(top_mlp_in)
        return output