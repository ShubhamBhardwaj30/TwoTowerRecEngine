import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, post_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.post_tower = nn.Sequential(
            nn.Linear(post_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, user_x, post_x):
        user_emb = self.user_tower(user_x)
        post_emb = self.post_tower(post_x)
        scores = (user_emb * post_emb).sum(dim=1)  # raw logits
        return scores
    


