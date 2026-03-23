# Module 6: Algorithm Cheat Sheet (Master Reference)

This is the "Emergency Survival Guide" for the Meta MLE loop. It summarizes everything we've built over the last 5 modules.

***

## 1. Linear / Logistic Regression
*   **Mechanism:** Weighted Sum of Features + Bias. Learns the plane that separates classes or fits the continuous trend.
*   **Loss Function:**
    *   Linear: **Mean Squared Error (MSE)**
    *   Logistic: **Log-Loss (Binary Cross-Entropy)**
*   **When to Use:** Baseline models, simple data, or when extreme interpretability is needed.
*   **Hyperparameters:**
    *   `C` (Regularization strength): Small `C` = Stronger penalty (Underfitting); Large `C` = Weaker penalty (Overfitting).
    *   `penalty` (`l1` vs `l2`): `l1` for feature selection, `l2` for stability.
*   **Gotchas:**
    *   **Multicollinearity:** Deeply hurt by correlated features.
    *   **Outliers:** Linear models are physically "pulled" by distant outliers.
    *   **Scaling:** Absolute necessity if Regularization is applied.

## 2. Decision Trees (Vanilla)
*   **Mechanism:** Recursive partitioning. Splits data on the "Best Feature" using Information Gain / Gini Impurity.
*   **Loss Function:** **Gini Impurity** or **Entropy** (for classification), **MSE** (for regression).
*   **When to Use:** Simple problems, understanding features.
*   **Hyperparameters:**
    *   `max_depth`: Limits the hierarchy to prevent overfitting.
    *   `min_samples_split`: Forces a split only if it includes a significant number of people.
*   **Gotchas:**
    *   **Inherent Bias:** A single tree is a high-variance model. It will memorize your dataset if the depth is infinite.

## 3. Random Forest (Bagging)
*   **Mechanism:** Parallely trains $N$ deep trees on random subsets of **Rows** and **Columns**. It "Averages" their votes to decide.
*   **Loss Function:** **Gini Impurity** or **Entropy** (Inherited from the base trees).
*   **When to Use:** General non-linear problems, robust defaults.
*   **Hyperparameters:**
    *   `n_estimators`: Total trees (usually 100+). More is better but slower.
    *   `max_features`: How many features to sample for each split.
*   **Gotchas:**
    *   **Slow Inference:** You have to run the data through 1,000 trees simultaneously.
    *   **OOB Error:** "Out-of-Bag" error is a free way to validate without a test set.

## 4. XGBoost / Gradient Boosting (Boosting)
*   **Mechanism:** Sequentially trains trees. Each new tree focuses **only on the errors** of the previous trees (Residuals).
*   **Loss Function:** **Binary Cross-Entropy** (Log-Loss), **MSE**, or any differentiable custom loss function.
*   **When to Use:** Competitive modeling, most tabular business problems.
*   **Hyperparameters:**
    *   `learning_rate` (eta): Shrinks the influence of each individual tree to prevent "rushing" into overfitting.
    *   `subsample` / `colsample`: Regularization by row/column sampling.
*   **Gotchas:**
    *   **Fragile Tuning:** It overfits much faster than Random Forest if not regularized.
    *   **Categoricals:** Older versions require One-Hot Encoding (OHE).

## 2. Support Vector Machines (SVM)
*   **Mechanism:** Finds the **Hyperplane** that creates the **Maximum Margin** between support vectors.
*   **Loss Function:** **Hinge Loss** (Penalizes points that are on the wrong side of the margin).
*   **When to Use:** Small datasets with high dimensions (e.g., text categorization).
*   **Hyperparameters:**
    *   `kernel` (`poly`, `rbf`): Projects data into higher dimensions where a linear split is possible.
    *   `C`: The trade-off between violating the margin vs creating a simple boundary.
*   **Gotchas:**
    *   **The Scaling Mandate:** Distance-based; scaling is a life-or-death requirement.
    *   **Computation:** $O(n^2)$ complexity; do not use for 1M+ rows.

## 6. K-Nearest Neighbors (KNN)
*   **Mechanism:** "Lazy Learner." It doesn't learn a model; it just stores the data and finds the "closest" $K$ neighbors for every new point.
*   **Loss Function:** None (Uses **Distance Metrics** like Euclidean or Manhattan during inference).
*   **When to Use:** Clustering, small simple search tasks.
*   **Hyperparameters:**
    *   `K`: Number of neighbors. Small $K$ is noisy; Large $K$ is smooth.
*   **Gotchas:**
    *   **The Curse of Dimensionality:** In high dimensions, every point becomes "far" from every other point, making distance meaningless.
    *   **Scaling:** Crucial.

## 7. Multi-Layer Perceptron (Neural Net)
*   **Mechanism:** Layers of interconnected "active" nodes. Uses **Backpropagation** and **Stochastic Gradient Descent (SGD)** to minimize loss.
*   **Loss Function:** 
    *   Binary Classification: **Binary Cross-Entropy (BCE Loss)**
    *   Multi-class Classification: **Categorical Cross-Entropy (CCE Loss)**
    *   Regression: **Mean Squared Error (MSE)**
*   **When to Use:** Large data, complex non-tabular data (Images, Text, Embeddings).

*   **Hyperparameters:**
    *   `hidden_layer_sizes`: Complexity of the architecture.
    *   `activation` (ReLU, Sigmoid): The "Activation Function" that introduces non-linearity.
*   **Gotchas:**
    *   **Data Hungry:** Requires massive amounts of data to beat a Random Forest.
    *   **Convergence:** High risk of getting stuck in a local minima if learning rate is wrong.

***

## 8. Naive Bayes (Probabilistic)
*   **Mechanism:** Uses Bayes' Theorem to calculate the probability of each class based on feature evidence. It "naively" assumes all features are independent.
*   **Loss Function:** Effectively **Log-Likelihood** maximization.
*   **When to Use:** Spam filtering, text classification, simple baselines.
*   **Gotchas:** Independence assumption is often violated, and it can't capture feature interactions.

## 9. DBSCAN (Density-Based Clustering)
*   **Mechanism:** Groups points that are density-reachable from each other.
*   **Loss Function:** None (Density-based search).
*   **When to Use:** Spatial data, datasets with noise/outliers, and clusters of non-spherical shapes.
*   **Gotchas:** Struggles with clusters of varying densities and high-dimensional data.

