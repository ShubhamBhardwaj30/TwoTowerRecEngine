# Meta Recommendation Systems: Master Study Guide

Welcome to the complete MLE Manager preparation vault for Meta's Recommendation Systems. This architecture represents the "First Principles" of how platforms like Instagram, Facebook, and Threads filter 1 Billion items down to a 10-item feed in under 200 milliseconds.

### Table of Contents

1.  **[Glossary: Jargon & Concept Reference](./00_jargon_glossary.md)**
    *   *Start here if a mathematical or architectural acronym is unclear.*
2.  **[Stage 0: Problem Understanding & Business Goals](./01_problem_understanding.md)**
    *   *Optimizing for long-term retention, the Multi-Sided Marketplace, and the MSI Formula.*
3.  **[Stage 1: Retrieval (Candidate Generation)](./02_retrieval_and_faiss.md)**
    *   *Two-Tower embeddings, FAISS, Inverted File Index (IVF), Product Quantization (PQ), and the Lambda Data Architecture.*
4.  **[Stage 2: Filtering (The Rules Engine)](./03_filtering_rules.md)**
    *   *Separation of Concerns and deterministic policy execution.*
5.  **[Stage 3: Ranking (DLRM) & Precision Scoring](./04_ranking_and_dlrm.md)**
    *   *Feature Disentanglement, Bottom MLPs, and the Dot-Product Interaction Layer.*
6.  **[Stage 4: Re-Ranking, Calibration, & BO](./05_reranking_and_calibration.md)**
    *   *Bayesian Optimization, Diversity routing, and final feed pacing.*
