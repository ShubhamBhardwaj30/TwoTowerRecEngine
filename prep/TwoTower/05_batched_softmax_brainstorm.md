# Brainstorming: In-Batch Softmax for Two-Tower Models

Instead of point-wise Binary Cross-Entropy (BCE), using "Batched Softmax" (In-Batch Negatives) is a highly efficient way to train retrieval models.

## 1. The Core Mechanism

Currently, the model calculates one similarity score per $(\text{user, post})$ pair and uses BCE to classify it as $0$ or $1$.

**With In-Batch Softmax:**
For a batch of size $N$, we compute a **Similarity Matrix** $S$ of size $N \times N$:
$$ S = U \cdot P^T $$
Where:
- $U \in \mathbb{R}^{N \times D}$ (User Embeddings)
- $P \in \mathbb{R}^{N \times D}$ (Post Embeddings)
- $S_{ij}$ is the cosine similarity (or dot product) between user $i$ and post $j$.

The **Loss Function** treats the diagonal elements $S_{ii}$ (where the user $i$ is paired with the positive post $i$) as the "Target Class." All other $N-1$ elements in the row are treated as **Implicit Negatives**.

## 2. Why it’s better

1.  **Explosive Negative Sampling**: Instead of sampling 1 negative per positive, a batch of 512 gives you **511 negatives per positive** for "free."
2.  **GPU-Friendly**: Matrix multiplication ($U P^T$) is extremely optimized for CUDA/MPS. Calculating $N^2$ scores is faster than calculating $N$ scores in a loop.
3.  **Gradient Information**: Every user in the batch provides a gradient update against every post in the batch. This results in much smoother and faster convergence.
4.  **No Sampling Required**: You don't need a separate "Negative Sampling" step in the data loader, which often becomes a bottleneck.

## 3. The Implementation (Conceptual)

Instead of `(N,)` logits, the model returns an `(N, N)` score matrix.

```python
# 1. Dot-product for the whole batch
scores = torch.matmul(user_embeddings, post_embeddings.T) # (N, N)

# 2. Add a Temperature Parameter (tau)
# High tau: Softmax is "Flat". Low tau: Softmax is "Peaky".
tau = 0.07 
scores = scores / tau

# 3. Targets are the indices 0, 1, 2, ..., N-1
targets = torch.arange(batch_size).to(device)

# 4. Cross-Entropy Loss
loss = nn.CrossEntropyLoss()(scores, targets)
```

## 4. Challenges & Tweaks

-   **Temperature Scaling ($\tau$)**: This is the most critical hyperparameter. If $\tau$ is too high, the model doesn't care about the negatives. If it's too low, it's too sensitive to noise.
-   **Batch Size Dependency**: This loss is biased by the batch size. Smaller batches might not provide enough "Hard Negatives." Modern industrial systems use batch sizes of 4096+ or "Streaming Negatives" (MoCo-style) to decouple negative count from batch size.
-   **Selection Bias**: If multiple users in the batch clicked the *same* post, you must handle the accidental "collisions" where row $i$ and row $k$ both share the same label (Positive Masking).

***

## 5. Current Implementation

As of the latest update, the Two-Tower pipeline uses **In-Batch Softmax** as the primary training objective.

### Model Changes (`src/train/two_tower.py`)
The `forward` method now returns the raw **User and Post Embeddings** instead of a single dot-product score. This modularity allows the trainer to handle the loss calculation using the full batch context.

### Trainer Changes (`src/train/two_tower_trainer.py`)
1.  **Mini-Batching**: Introduced `DataLoader` and `TensorDataset` to process interactions in chunks of **512**.
2.  **Positive Filtering**: The trainer filters the dataset to focus on positive interactions (`label=1`) for the contrastive loss.
3.  **Temperature Scaling**: A temperature of **$\tau = 0.07$** is applied to the similarity matrix to sharpen the softmax distribution and focus on hard negatives.
4.  **Loss Function**: Switched from `BCEWithLogitsLoss` to `CrossEntropyLoss` against the diagonal of the similarity matrix.
5.  **LogQ Correction**: To prevent popularity bias, the trainer now subtracts $\log(Q(i))$ (item sampling probability) from the similarity scores before computing the loss. This ensures that popular items are not unfairly penalized for frequently appearing as negatives.
