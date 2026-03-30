# Ranker Architectural Deep-Dive: Stage-1 to Stage-2 Integration

This document provides a technical walkthrough of how the Stage-2 DLRM Ranker utilizes the embeddings produced by the Stage-1 Two-Tower Retrieval model.

## 1. Data Flow & Tensor Construction
The ranking model does not just look at cold IDs; it leverages the geometric semantic space learned during the retrieval stage. In the training pipeline, we construct a 3-way input:

1.  **Dense Path:** Continuous features (Age, counts) $\rightarrow$ Bottom MLP $\rightarrow$ 64D Vector.
2.  **Sparse Path:** Categorical IDs (User, Post) $\rightarrow$ Hashing Embedding $\rightarrow$ 64D Vector.
3.  **Tower Path:** Pre-computed Two-Tower Embeddings $\rightarrow$ Passthrough $\rightarrow$ 64D Vector.

## 2. The "Double Embedding" Rationale
We use two different types of embeddings for the same entities (Users and Posts):
| Feature Type | Source | Purpose |
| :--- | :--- | :--- |
| **Two-Tower Embedding** | Pre-calculated (Stage 1) | Captures **General Semantic Similarity** and community context. |
| **Hashing Embedding** | Trainable (Stage 2) | Captures **ID-Specific Memorization** and fine-grained patterns. |

By using both, the ranker can balance whether a user "usually likes sports" (Two-Tower) against whether they "specifically like this post" (Hashing).

## 3. Code Breakdown: `ranker_nn.py`
The implementation in `src/train/ranker_nn.py` handles the interaction as follows:

```python
# Extract Tower signals (already 64D)
tt_u_emb = tower_x[:, 0, :]
tt_p_emb = tower_x[:, 1, :]

# Stack all vectors for interaction (Total 7 vectors)
all_vectors = torch.stack([
    bottom_out,   # Dense Context
    u_emb, p_emb, # Hashing Identities
    type_emb,     # Categorical Type
    hour_emb,     # Categorical Time
    tt_u_emb,     # Stage-1 User Semantic
    tt_p_emb      # Stage-1 Post Semantic
], dim=1)

# Pairwise Dot-Products (The Interaction Matrix)
interactions = torch.bmm(all_vectors, all_vectors.transpose(1, 2))
```

## 4. Why Pairwise Dot-Products?
The "Stage 3" Ranker is designed for **Precision**. By calculating the dot product between the Two-Tower semantic vector and the Hashing identity vector, the model explicitly learns how the "Semantic Space" and "ID Space" correlate. 

This results in **21 unique interaction scores** that are fed into the final Top MLP, allowing it to make a high-precision multi-task prediction for Likes, Comments, and Shares.
