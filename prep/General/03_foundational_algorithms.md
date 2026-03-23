# Module 3: Foundational Algorithms

Understanding the mathematical DNA of different algorithms allows you to choose the "right tool for the job" based on the data's shape and the business's need for interpretability.

***

## 1. Linear & Logistic Regression (The Linear Benchmarks)
Linear models represent the simplest relationship: a weighted sum of inputs plus a bias.
*   **Key Assumption:** They assume a linear relationship between features and the target (or the log-odds of the target).
*   **Multicollinearity:** These models are highly sensitive to correlated features. If two features are nearly identical, the model's coefficients ($weights$) become unstable and lose interpretability.
*   **Handling Non-Linearity:** To capture non-linear patterns (like a U-shaped curve), you must manually engineer **Polynomial Features** (e.g., $x^2$) or use **Interaction Terms**.

## 2. Tree-Based Models (The Workhorses)
Decision Trees, Random Forests, and Gradient Boosted Trees (XGBoost/LightGBM) work by recursively partitioning the feature space.
*   **Non-Linearity:** They capture complex, non-linear relationships natively. Each "split" in a tree is a step-function, which can approximate any curve.
*   **Robustness:** They are mostly immune to outliers and **do not require feature scaling**. Since they only care about whether a value is "greater than" or "less than" a threshold ($x > 5.5$), the raw magnitude of the feature doesn't change the decision boundary.
*   **Correlation:** Random Forests handle multicollinearity better because they randomly sample a subset of features for every split, preventing one dominant feature from overshadowing its correlated peers.

***

## 3. Distance-Based models (KNN, SVM, K-Means)
These models rely on the geometric distance (usually Euclidean) between points in a multi-dimensional space.
*   **The Scaling Mandate:** Because they calculate distances, **Feature Scaling is Mandatory**. If "Income" ranges from 0 to 500,000 and "Age" ranges from 0 to 100, the Income will mathematically drown out the Age. A move of 1 year in age will be dwarfed by a move of \$1 in income in the Euclidean distance formula ($d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$).
*   **K-Means vs. Hierarchical Clustering:** 
    *   **K-Means:** Requires you to specify $K$ (number of clusters) upfront. Fast and scalable.
    *   **Hierarchical (Agglomerative):** Builds a "Dendrogram" (tree of clusters) from the bottom up. Does NOT require $K$ upfront. Useful for seeing the "nested" structure of your data.

***

## 4. Probabilistic Models (Naive Bayes)
*   **Mechanism:** Based on **Bayes' Theorem**. It calculates the probability of a class given the features by assuming all features are **independent** (the "Naive" assumption).
*   **When to Use:** Text classification (Spam detection), very large datasets where speed is critical, and as a baseline for high-dimensional data.
*   **Gotchas:** The independence assumption is almost always false in the real world, but the model still performs surprisingly well, especially for NLP tasks.


---

## 🚀 Deep Dive: Operational Trade-offs

### Q: Why do trees ignore scaling while KNN requires it?
*   **Mathematical Intuition:** A Tree splits data based on **Rank** (Is $X > 50$?). Scaling $X$ by 1,000 doesn't change the relative rank—every point >50 is still >50 relative to its neighbors. However, **KNN** is based on **Magnitude** (How far is $X$ from $Y$?). Without scaling, the feature with the largest raw range (e.g., Millions vs Units) dominates the distance calculation, making the smaller-range features invisible to the model.

### Q: How do you force a Linear Model to see a Non-Linear curve?
*   **The Engineering Trick:** If you have a U-shaped relationship ($Y$ is high at $X$=0 and $X$=100, but low in the middle), you cannot fit a line to it. To solve this in **Logistic Regression**, you create **Polynomial Features**. By adding $X^2$ as a new input column, you mathematically lift the 2D "curve" into a 3D space where a flat plane (a linear boundary) can suddenly slice through it perfectly.
