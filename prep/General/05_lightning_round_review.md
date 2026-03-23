# Lightning Round: Interview Review

This document archives the "First Principles" mock interview questions and the expert-level answers for final review before the Meta MLE loop.

***

### Q1: [Paradigms] Grouping 1M Users into 5 Personas
*   **Answer:** **K-Means Clustering**.
*   **Why:** It is an Unsupervised algorithm designed for segmentation. It finds natural centers (centroids) in the behavioral data to minimize the distance between users in the same group.

### Q2: [Data Prep] Handling a 1-5 Star Rating column
*   **Answer:** Treat as **Numerical/Ordinal**, NOT Categorical.
*   **Trade-off:** If you One-Hot Encode ($1,0,0,0,0$), the model treats the difference between 1-star and 2-stars the same as the difference between 1-star and 5-stars. By keeping them numerical, you preserve the natural rank order.

### Q3: [Algorithms] Scaling Height and Weight
*   **Answer:** Scale for the **SVM**, but NOT for the **Random Forest**.
*   **Why:** SVM uses Euclidean distance (geometry); without scaling, the larger magnitude feature (Height in cm) will drown out the smaller one (Weight in kg). Forests use binary threshold splits (Rank) which are magnitude-independent.

### Q4: [Metrics] AUC of 0.51 and 99% Recall
*   **Answer:** **Poor Performing Model**.
*   **Why:** An AUC of $0.51$ is a random guesser (a coin-flip). If the Recall is $99\%$, it means the model is likely just predicting "Positive" for almost every case to "catch" the targets without actually understanding the patterns.

### Q5: [Case Study] Weekend Error Spikes
*   **Answer:** **Time-Blindness / Lag**.
*   **The Fix:** Engineered Feature: **`is_weekend`**. The model currently doesn't know what day of the week it is. By explicitly adding the day type, the model can shift its decision boundary to account for different behavioral patterns on Saturdays/Sundays.
