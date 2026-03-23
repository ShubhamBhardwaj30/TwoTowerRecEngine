# Stage 3: Ranking (Deep Learning Recommendation Model)

**Goal:** Take the ~800 safe candidates from Stage 2 and use heavy ML to generate ultra-precise predicted interaction probabilities ($P(\text{Like}), P(\text{Comment})$). Prioritize **Precision** over Recall.

## 1. Feature Disentanglement
Unlike the Two-Tower model which squashes every feature into one vector, a Ranker requires massive fidelity.
*   **The Problem:** Over-simplifying features. If `User_Age` is baked into the permanent `UserID` embedding, the system has a hard time reacting dynamically as the user grows older or generating specific cross-features (e.g., this user is 16 AND this specific post is violent).
*   **The Principle:** We disentangle the static **Identity** (Collaborative IDs) from the dynamic **Context** (Time, Age, Session details) at the input layer.

## 2. DLRM Architecture
Meta invented DLRM to handle these vastly differently feature types efficiently.

*   **The Sparse Path (Categorical):** High-cardinality IDs (`UserID`, `PostID`) are fed into Embedding Tables to output a standard vector (e.g., 64D). 
    *   *Optimization:* We use Hashing to bound the size of these tables in RAM, using Bayesian Optimization to tune the collision vs memory tradeoff. 
*   **The Dense Path (Continuous):** Scalars (`Age`, `Historical_CTR`) are fed into a **"Bottom MLP"**. 
    *   *Optimization:* The Bottom MLP transforms numbers into 64D Vectors. This creates a "Level Playing Field," ensuring low-dimensional scalars are not mathematically overshadowed by high-dimensional embeddings during training.

## 3. The Dot-Product Interaction Layer
Once all Sparse identities and Dense contexts are 64D vectors, they meet in the middle.
*   **Concatenation (The Flawed Default):** If you stack 64D vectors together (`np.hstack`), the final MLP must manually learn how to compare values weight-by-weight across epochs.
*   **Dot-Product (The Shortcut):** DLRM explicitly calculates the **Dot-Product (Geometric Similarity/Correlation)** between every pair of features (e.g., $User_{64D} \cdot Post_{64D}$). 
*   **Why it works:** It hands the final Top MLP the exact mathematical correlation ("Similarity Score") of every interaction, drastically speeding up convergence and accuracy.
