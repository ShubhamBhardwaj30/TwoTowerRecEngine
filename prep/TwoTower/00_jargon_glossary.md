# Recommendation Systems: Jargon & Glossary

This is your master reference for the complex terminology expected in an MLE Manager interview.

## Core Architecture
*   **DLRM (Deep Learning Recommendation Model):** Meta's open-source neural network architecture specifically designed to handle both sparse (categorical IDs) and dense (continuous scalars) data efficiently.
*   **Two-Tower Model:** A neural network with two separate pathways (User Tower and Item Tower) that individually encode features into vectors before comparing them. Primarily used in the Retrieval (Stage 1) phase.
*   **Lambda Architecture:** A data processing pattern that uses both a Slow Path (Batch/Hadoop for massive historical data) and a Fast Path (Streaming/Flink for real-time recent clicks).

## Scaling & Constraints
*   **The Cold Start Problem:** The inability of classic collaborative filtering (like Matrix Factorization) to recommend an item that has 0 previous engagements. Solved by Neural Networks using raw content features.
*   **Curse of Dimensionality:** A mathematical phenomenon where distances between points become meaningless in high-dimensional spaces (e.g., 64D or 384D). Forces systems to use approximations (FAISS) instead of exact spatial trees (Geohashes).

## Approximate Nearest Neighbors (ANN) & FAISS
*   **FAISS (Facebook AI Similarity Search):** An open-source library for efficient similarity search of dense vectors, designed for billion-scale databases.
*   **IVF (Inverted File Index):** An ANN technique that runs K-Means clustering offline to group vectors into "Voronoi Cells." At inference, the system only searches the cells closest to the user.
*   **PQ (Product Quantization):** A mathematical compression technique. It splits high-dimensional vectors into chunks, maps them to short integer IDs (Codebooks), and shrinks RAM usage by ~90%.
*   **ADC (Asymmetric Distance Computation):** An optimization tied to PQ. The user vector remains uncompressed (exact) while the database vectors remain compressed. FAISS calculates the math directly against the look-up IDs, entirely bypassing the need to "decompress" the database into RAM during a search.

## Optimization Strategy
*   **MSI (Meaningful Social Interaction):** A weighted metric formula prioritizing community-building actions (long comments, shares) over passive grazing (scrolls, simple likes).
*   **BO (Bayesian Optimization):** A hyperparameter tuning technique using a Gaussian Process surrogate model. It efficiently searches for the "Pareto Optimal" balance between competing metrics (Cost vs Accuracy, Engagement vs Integrity) without requiring millions of brute-force experiments.
