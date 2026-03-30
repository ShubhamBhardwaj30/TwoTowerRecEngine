# Stage 1: Retrieval (Candidate Generation)

**Goal:** Filter 1 Billion candidates down to ~1,000 in under 50ms. Prioritize **Recall** (don't miss anything relevant) over Precision.

## 1. Data Prep & The Lambda Architecture
Data must be fresh to be relevant. We cannot rely strictly on overnight batches.
*   **Slow Path (Batch):** Calculates high-volume historical aggregates (e.g., `user_30_day_like_ratio`). This information is highly stable.
*   **Fast Path (Stream):** Ingests live actions (e.g., `user_just_hid_basketball_video`). Ensures the model is reactive to session state. 
*   *Note: Streaming data is volatile and must be normalized (Z-score/Quantiles) so it does not overpower the stable, historical baseline.*

## 2. Model Selection: Why Two-Tower over Matrix Factorization?
*   **Matrix Factorization Limitations:** $O(N)$ dot-product at inference takes minutes. Cannot handle **Cold Start** (a post with 0 clicks has no identity in the matrix).
*   **The Two-Tower Advantage:** Encodes raw *features* (text, age, hashtags) into the User Tower and Post Tower. It instantly generates embeddings for brand-new posts, skipping the Cold Start trap.

## 3. The Inference Solution: FAISS & ANN
Even with Two-Tower embeddings, doing a real-time dot-product of 1 User vs 1 Billion Posts is computationally impossible inside 50ms. We use Approximate Nearest Neighbors (ANN) via FAISS.

1.  **IVF (Inverted File Index) - The Compute Fix:** 
    *   Offline, we run K-Means to cluster 1B posts into 1 Million "Voronoi Cells" (neighborhoods).
    *   Online, instead of searching 1B posts, we check the User Vector against the 1 Million cell Centroids, and only search the closest buckets. Math drops from $O(N)$ to $O(\sqrt{N})$.
2.  **PQ (Product Quantization) - The RAM Fix:** 
    *   1 Billion 64D vectors = ~256GB RAM. Disk I/O is too slow for real-time reads. Fast retrieval demands keeping everything in RAM.
    *   PQ mathematically compresses the post vectors by slicing them into 8 bits and mapping them to "Codebook IDs". Size drops by ~90% (fits in <10GB RAM).
3.  **ADC (Asymmetric Distance Computation) - The Speed Fix:** 
    *   FAISS *never decompresses* the PQ vectors at inference time. It creates a rapid-lookup table against the uncompressed User Vector and performs addition directly against the PQ Codebook IDs.

### Expert Q&A: Geospatial vs FAISS
**Q: Why do we use FAISS instead of Exact Trees (like QuadTrees) if accuracy matters?**
**A:** Because of the **Curse of Dimensionality.** Geospatial structures (Geohashes, S2) are mathematically perfect for 2D maps because 2-dimensional boundaries are cheap to traverse. The Two-Tower creates a **64D to 384D** latent space. In hyper-dimensional space, exact trees break down completely, forcing us to use probabilistic approximations (FAISS/PQ).

***

## 4. Training Strategy: In-Batch Softmax & LogQ Correction

To maximize recall at scale, the model is trained using **In-Batch Softmax** (In-Batch Negatives). For every positive pair $(u, i)$ in a batch of size $N$, the other $N-1$ items in the batch act as negatives.

### The Popularity Bias Problem
In a random batch, popular items appear as negatives much more frequently than niche items. This means the model is penalized disproportionately for being "similiar" to popular items, simply because it sees them as negatives more often.

### The Fix: LogQ Correction
To solve this, we apply **LogQ Correction** to the similarity scores ($s$) before the loss calculation:
$$ s_{corrected}(u, i) = s(u, i) - \log(Q(i)) $$
- **$Q(i)$**: The probability of post $i$ being sampled in the batch (its frequency in the data).
- **Result**: Popular items (High $Q$) receive a larger penalty, while niche items (Low $Q$) get a mathematical "boost." This prevents "Popularity Bias" from dominating the semantic latent space.

### Example: Popular vs. Niche Post
Suppose we have two posts in a batch:
1.  **Post A (Popular)**: $Q(A) = 0.9$ (90% of training data) $\rightarrow \log(0.9) \approx -0.1$
2.  **Post B (Niche)**: $Q(B) = 0.01$ (1% of training data) $\rightarrow \log(0.01) \approx -4.6$

If the raw similarity score for both items is $10.0$:
- **Post A (Corrected)**: $10.0 - (-0.1) = 10.1$
- **Post B (Corrected)**: $10.0 - (-4.6) = 14.6$

The niche Post B receives a significant numerical "boost" ($+4.5$), making it much more likely to be selected than the over-exposed Post A, even though their raw semantic similarities were identical.

***

### Q&A: Won't LogQ Correction "Kill" My Viral Posts?

**Q: If a post is globally popular (viral), doesn't LogQ correction penalize it so much that it never reaches the user?**

**A: Yes, but only in the Retrieval Stage—and that is actually a good thing!**

1.  **Retrieval is about Taste, not Trend**: The Two-Tower model's job is to find items that match the user's **Specific Interest** (e.g., Chess, Cooking). If a post is viral (e.g., a massive celebrity news story), it will be similar to *everyone* in the batch. LogQ correction "removes" that popularity noise so the model can see if the post actually matches the user's **semantic** interest.
2.  **The Ranker (Stage 3) is the Trend-Maker**: We *want* viral posts to be shown, but we don't want the **Retriever** to be a "Popularity Engine." 
    - The **Retriever** finds posts that match the user's taste.
    - The **Ranker** (DLRM) includes a feature for `item_global_popularity`.
    - Once the Retriever has found a broad set of candidates, the Ranker looks at them and says, *"Out of these 1,000 posts that Match your taste, this one is also Viral, so I'll put it at position #1."*

This separation ensures your feed is **Personalized** AND **Fresh**, rather than just being a list of the top 10 most popular posts for everyone.
