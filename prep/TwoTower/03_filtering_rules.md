# Stage 2: Filtering (The Rules Engine)

**Goal:** Clean the 1,000 FAISS candidates generated from Retrieval, dropping unsafe, illegal, or redundant posts before they reach the expensive Ranking model.

## 1. Separation of Concerns
A common junior mistake is assuming the Ranking Neural Network should handle everything ("If the post is bad, the network will just score it a 0.001").
*   Using a heavy Deep Learning Recommendation Model (DLRM) to enforce business rules is like **"using a sword where you need a needle."**
*   The Ranker's ONLY job should be predicting probabilistic engagement. It should not be burdened with Meta's Safety and Integrity logic.

## 2. Determinism vs. Probability
*   Neural Networks are probabilistic. 
*   If Community Guidelines dictate that a blocked author, repeated misinformation, or borderline NSFW content must not be shown to a minor, we cannot rely on a 99% probability. We need a 100% hard rule.

## 3. The Implementation
Because this stage must be lightning fast, it relies on strict heuristics:
*   **Bloom Filters:** Used to check if the user has already seen this post in the last 24 hours (Deduplication). Bloom Filters are incredibly fast, memory-efficient probabilistic data structures perfectly suited for "seen" states.
*   **Fast KV Lookups:** Checking the `author_id` against a cached `blocked_users_list`.
*   **Pre-computed Flags:** Checking if the post was flagged by offline Safety/Integrity classifier jobs that run when the post is uploaded.
