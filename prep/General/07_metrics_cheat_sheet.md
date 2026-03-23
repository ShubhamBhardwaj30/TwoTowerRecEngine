# Module 7: Metrics Cheat Sheet (Master Reference)

This sheet determines which "Y-axis" you use to prove your model is succeeding. Interviewers will often ask: *"How do you know if this model is actually solving the business problem?"*

***

## 📊 Classification (Categorical Predictions)
Use these when your output is a class (e.g., Churn vs. No Churn).

| Metric | Best For... | Mathematical Intuition |
| :--- | :--- | :--- |
| **Accuracy** | Balanced Data | (Correct / Total). Only use if classes are $50/50$. |
| **Precision** | Cost of False Positives | $TP / (TP + FP)$. "How often am I right?" |
| **Recall** | Cost of False Negatives | $TP / (TP + FN)$. "How many did I catch?" |
| **F1-Score** | Imbalanced Data | $2 \times \frac{P \times R}{P + R}$ (The "Truthful" balance). |

| **ROC-AUC** | Discrimination | Power to rank a positive sample higher than a negative one. |

***

### 🧱 The Confusion Matrix (Truth Table)
Use this to build almost any classification metric.

| | **Actual Positive** | **Actual Negative** |
| :--- | :--- | :--- |
| **Pred Positive** | **TP** (True Positive) | **FP** (Type I Error) |
| **Pred Negative** | **FN** (Type II Error) | **TN** (True Negative) |
| **Log-Loss** | Confidence | Penalizes "Sure but wrong" predictions heavily. Uses the raw probability. |

---

## 📈 Regression (Continuous Numbers)
Use these when your output is a price, a days-to-event, or a score.

| Metric | Best For... | Mathematical Intuition |
| :--- | :--- | :--- |
| **MAE** | Typical Error | Mean Absolute Error. "On average, I'm off by \$X." Robust to outliers. |
| **MSE** | Penalizing Big Errors | Squares the error. If you miss by 10, the penalty is 100. Use to avoid catastrophic failures. |
| **RMSE** | Interpretation | Root Mean Square Error. Keeps the " penalty" in the same units as the target. |
| **R-Squared** | Explained Variance | How much of the data's "wiggle" does your model explain? |

---

## 👥 Clustering (Unsupervised Groups)
Use these when there are no ground-truth labels.

| Metric | Best For... | Mathematical Intuition |
| :--- | :--- | :--- |
| **Inertia** | Finding K | Sum of squares within a cluster. Lower is better (more compact). |
| **Elbow Method** | Selection | Plotting Inertia vs K. The "bend" shows the point of diminishing returns. |
| **Silhouette** | Separation | Measures if points are closer to their own cluster than the nearest neighbor. |

---

## 🏆 Ranking (Recommender Systems)
Use these for sorted lists, feeds, and search results.

| Metric | Best For... | Mathematical Intuition |
| :--- | :--- | :--- |
| **NDCG** | Scored Lists | Normalized Discounted Cumulative Gain. Penalizes relevant items appearing lower in the feed ($\log$ discount). |
| **MRR** | Search/Lookup | Mean Reciprocal Rank. "Where did the first relevant result appear?" ($1/rank$). |
| **Precision@K** | "Above the fold" | Of the first $K$ items shown, what \% were relevant? |
| **Hit Rate** | Conversion | Did the target item appear in the top $K$ at all? (Binary $1/0$). |
