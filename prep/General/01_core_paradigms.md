# Module 1: Core ML Paradigms

## 1. Supervised Learning
**The Core Requirement:** Supervised learning absolutely requires **Labels** (Ground Truth). 
You cannot train a supervised model on raw browser events unless you mathematically extract a targeted "Output Column" ($Y$). 
*   **Mechanism:** It learns the mapping function $f(X) = Y$ using historical examples so that when we see a new $X$, we can predict its $Y$.
*   **Examples:** "Did the user click?" ($1$ or $0$), "How much will this house sell for?" ($\$500k$).

## 2. Unsupervised Learning
**The Core Requirement:** Unsupervised learning uses **Unlabeled Data**. There is no target variable, no output column, and no "Right Answer."
*   **Mechanism:** It looks strictly at the raw input features ($X$) to discover hidden structures, groupings, or mathematical anomalies without a specific end-goal in mind.
*   **Examples:** 
    *   **Clustering (K-Means):** Grouping overlapping users into 5 distinct "Buyer Personas" based on behavioral similarity.
    *   **Dimensionality Reduction (PCA):** Compressing 5,000 sparse tracking features into 50 dense core components while retaining 95% of the statistical variance.

## 3. Reinforcement Learning
**The Core Requirement:** An **Agent** interacting with an **Environment** to maximize a **Reward**. There is no static historical dataset. The model generates its own data by trial and error.
*   **Examples:** Playing Chess, training self-driving cars, or dynamic pacing algorithms for ad-budgets.

---
### Real-World Execution (The First 7 Days)
When facing a completely raw 50TB database without any labels, the raw algorithm implementation is secondary to **Data Strategy**:
1. **Unsupervised First:** Run EDA and mathematical clustering on raw features to understand natural user segmentation and dataset quality before a target is even identified.
2. **Define the Supervised Label:** Sit with Product Management to define what the business actually cares about (e.g., Identifying the criteria for "Churn"). Feature Engineer the historical data to specifically manufacture a `has_churned` label column.
3. **Train Supervised Models:** Once the $Y$ column is defined, deploy regression or classification models on the $X$ features to predict it.

---
## 4. Classification vs Regression (Supervised Sub-Types)
When dealing with Supervised Learning (where the target $Y$ is formally defined), the algorithm architecture breaks down into two strictly different mathematical problems:

*   **Classification:** The output $Y$ is a **Discrete Category**. 
    *   *Example:* Predicting "Will the user churn?" (Output: `1` or `0`).
    *   *Mechanism:* The model outputs a bounded probability between 0 and 1 (often using a Sigmoid or Softmax activation), which is then mapped to exactly one rigid category constraint.
    *   *Multi-Class Variant:* Defining $Y$ into tiered discrete buckets (e.g., classifying users as Red, Yellow, or Green churn risk).
*   **Regression:** The output $Y$ is a **Continuous Number**.
    *   *Example:* Predicting "How many days until the user churns?" (Output: `14.5 days`).
    *   *Mechanism:* The model outputs an unconstrained, continuous value on an infinite scale.

### The Practitioner's Nuance: Probabilities vs Hard Labels
In real-world business scenarios, rigidly outputting a binary classification (`Churn = 1`) destroys valuable operational nuance. Instead, ML engineers often utilize **Probability Calibration** (treating the Classification algorithm's raw `0.0` to `1.0` output essentially like a Regression score). 
By outputting a raw $85\%$ risk factor rather than a hard `1`, you can correlate those continuous probabilities against actual "Days until Churn" to dynamically organize business interventions.
