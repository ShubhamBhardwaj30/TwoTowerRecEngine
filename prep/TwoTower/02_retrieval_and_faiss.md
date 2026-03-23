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
