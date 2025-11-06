import torch.nn as nn

class RankerNN(nn.Module):

    def __init__(self,input_dims, output_dims, hidden_dims=32, dropout=0.2):
        super().__init__()
        self.ranker = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dims, output_dims)
        )
        

    def forward(self, input_x):
        output = self.ranker(input_x)
        return output
    