import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from train.two_tower import TwoTowerModel
import torch.nn as nn

def test_softmax_loss():
    print("Testing In-Batch Softmax Loss...")
    batch_size = 4
    dim = 16
    hidden = 32
    
    model = TwoTowerModel(user_dim=dim, post_dim=dim, hidden_dim=hidden)
    user_x = torch.randn(batch_size, dim)
    post_x = torch.randn(batch_size, dim)
    
    # 1. Forward pass
    u_emb, p_emb = model(user_x, post_x)
    assert u_emb.shape == (batch_size, hidden)
    assert p_emb.shape == (batch_size, hidden)
    print("✓ Forward pass successful")
    
    # 2. Similarity Matrix
    temp = 0.07
    logits = torch.matmul(u_emb, p_emb.T) / temp
    assert logits.shape == (batch_size, batch_size)
    print("✓ Similarity matrix calculation successful")
    
    # 3. Loss Calculation
    criterion = nn.CrossEntropyLoss()
    targets = torch.arange(batch_size)
    loss = criterion(logits, targets)
    print(f"✓ Loss calculation successful: {loss.item():.4f}")
    
    # 4. Backward Pass
    loss.backward()
    print("✓ Backward pass successful")
    
    print("\nVerification PASSED!")

def test_logq_correction():
    print("\nTesting LogQ Correction...")
    batch_size = 4
    dim = 16
    hidden = 32
    
    model = TwoTowerModel(user_dim=dim, post_dim=dim, hidden_dim=hidden)
    user_x = torch.randn(batch_size, dim)
    post_x = torch.randn(batch_size, dim)
    log_q = torch.tensor([-1.0, -2.0, -3.0, -4.0]) # Example log probabilities
    
    u_emb, p_emb = model(user_x, post_x)
    logits = torch.matmul(u_emb, p_emb.T) / 0.07
    
    # Apply correction
    logits_corrected = logits - log_q.view(1, -1)
    
    assert logits_corrected.shape == (batch_size, batch_size)
    # Check if correction is applied to columns
    for i in range(batch_size):
        assert torch.allclose(logits_corrected[:, i], logits[:, i] - log_q[i])
    
    print("✓ LogQ correction applied correctly to columns")
    print("✓ Verification PASSED!")

if __name__ == "__main__":
    test_softmax_loss()
    test_logq_correction()
