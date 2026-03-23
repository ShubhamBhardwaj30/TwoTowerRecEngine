# Module 00: The Master ML Glossary

This document archives the mathematical "Definitions" that interviewers often use as a baseline test of your theoretical depth.

***

## 🧠 Information Theory & Trees
How models decide to split data into groups.

*   **Entropy ($H$):** Measures the amount of **Disorder** or **Uncertainty** in a dataset. If a group is $50/50$ churn/no-churn, entropy is high ($1.0$). If everyone is in one class, entropy is $0$.
    *   *Formula:* $H(S) = -\sum p_i \log_2(p_i)$
*   **Information Gain (IG):** The **Reduction in Entropy** achieved by splitting a dataset based on a specific feature.
    *   *Intuition:* "How much 'cleaner' did our data get after we separated people by Age?"
    *   *Formula:* $IG = Entropy(BeforeSplit) - Entropy(AfterSplit)$
*   **Gini Impurity:** Similar to Entropy but computationally faster (no $\log$). Measures the probability of a randomly chosen element being incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset.
    *   *Formula:* $Gini = 1 - \sum p_i^2$
    *   *Difference:* Gini favors larger partitions; Entropy favors smaller, more specific splits. (Both work similarly in practice).

---

## 📐 Distance & Probability Math
Foundations for KNN, Clustering, and Bayesian models.

*   **Euclidean Distance:** The straight-line distance between two points in $N$-dimensional space. (Used in KNN, K-Means).
    *   *Formula:* $d(\mathbf{p, q}) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}$
*   **Manhattan Distance (L1):** The "Taxicab" distance, moving only horizontally and vertically. (Used when features are not independent).
    *   *Formula:* $d(\mathbf{p, q}) = \sum_{i=1}^n |p_i - q_i|$
*   **Bayes' Theorem:** The rule for updating the probability of a hypothesis ($H$) based on new evidence ($E$).
    *   *Formula:* $P(H|E) = \frac{P(E|H) P(H)}{P(E)}$
    *   **Naive Bayes:** Predicts $Y$ by assuming $P(X_1, X_2...|Y) \approx P(X_1|Y)P(X_2|Y)...$

---


---

## ⚙️ Optimization & Math
How models actually "learn" their weights.

*   **Ordinary Least Squares (OLS):** The mathematical method used to find the "Line of Best Fit." It focuses on minimizing the sum of the squares of the vertical offsets (residuals).
    *   *Formula:* $Y = (X^T X)^{-1} X^T Y$ (Closed-form solution).
*   **Logistic Regression (The Logit):** Predicts the **Log-Odds** of an event. It maps any linear combination of features to a $0-1$ probability space using the Sigmoid.
    *   *Formula:* $\ln\left(\frac{p}{1-p}\right) = w_0 + w_1 x_1 + \dots$

*   **Relationship to Gradient Descent:**
    *   The **MSE (Mean Squared Error)** is the objective function for OLS.
    *   While OLS has a "Closed-form" mathematical formula (above), it involves inverting a massive matrix ($X^T X$), which is computationally impossible for millions of rows. 
    *   **Gradient Descent** is the iterative alternative. Instead of solving the equation in one step, we take tiny downhill steps (using derivatives) to find the same minimum point without the expensive matrix inversion.
*   **Loss Function ($L$):** Measures the error between the model's prediction ($\hat{y}$) and the true value ($y$).
*   **BCE Loss:** (Already documented above).
*   **F1 Score:** The Harmonic Mean of Precision and Recall.
    *   *Formula:* $F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
*   **Hinge Loss (SVM):** The loss function that penalizes a prediction only if it's on the wrong side of the margin or incorrectly classified.
    *   *Formula:* $L = \sum_{i=1}^n \max(0, 1 - y_i \hat{y}_i) + \lambda \|w\|^2$
*   **PCA Covariance:** Reduces dimensions by finding the eigenvectors of the data's covariance matrix.
    *   *Formula:* $\Sigma = \frac{1}{n-1} X^T X$ (Where $X$ is centered).
*   **Cost Function:** The average of the Loss Function across the entire training set.


*   **Gradient Descent:** The optimization algorithm that "walks" the model's weights $(\theta)$ downhill toward the minimum possible error.
    *   *The Step:* New\_Weight = Old\_Weight - Learning\_Rate $\times$ Gradient
*   **Learning Rate ($\alpha$ / $\eta$):** The size of the step in Gradient Descent.
    *   *Too High:* You overshoot the minimum.
    *   *Too Low:* It takes forever to learn.
*   **Backpropagation:** The algorithm used to calculate gradients effectively through a Neural Network by moving backward from the final error toward the input layers.

---

## 📊 Data Transformation
Preparing data for modeling.

*   **Min-Max Scaling (Normalization):** Rescales features to a fixed range, usually $[0, 1]$. Useful for algorithms that are not scale-invariant (e.g., K-Nearest Neighbors, Neural Networks).
    *   *Formula:* $X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$
*   **Standard Scaling (Standardization):** Rescales features to have a mean of $0$ and a standard deviation of $1$. Assumes data is normally distributed. Useful for algorithms that assume Gaussian distribution (e.g., Linear Regression, Logistic Regression).
    *   *Formula:* $X_{scaled} = \frac{X - \mu}{\sigma}$
*   **Robust Scaling:** Rescales data using the **Median** and the **Interquartile Range (IQR)**. This is the "Professional" choice when your data is messy and full of extreme outliers that would skew a standard Z-score.
    *   *Formula:* $X_{scaled} = \frac{X - Median}{Q_3 - Q_1}$ (where $Q_1$ and $Q_3$ are the 25th and 75th percentiles).

---

## 🏗️ Architecture & Generalization
The physical properties of your model.

*   **Sigmoid Function:** Squashes any real number into a probability range $[0, 1]$. Used for Binary Classification.
    *   *Formula:* $\sigma(x) = \frac{1}{1 + e^{-x}}$
*   **Softmax Function:** Generalizes Sigmoid to multiple classes. Ensures the sum of all class probabilities equals $1.0$, allowing the model to choose the "most likely" class among many.
    *   *Formula:* $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$
*   **Regularization:** Adding a penalty term to the loss function to prevent weights from becoming too large (Overfitting).
    *   **L1 (Lasso):** Absolutes. Drives weights to **zero**.
    *   **L2 (Ridge):** Squares. Shrinks weights to be **small**.
*   **XGBoost Objective Function:** The total function the model tries to minimize at every step of tree-building.
    *   *Formula:* $\text{Obj} = \sum_{i=1}^n L(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$
    *   *Regularization ($\Omega$):* $\Omega(f) = \gamma T + \frac{1}{2}\lambda \|\omega\|^2$ (where $T$ is the number of leaves and $\omega$ are the leaf scores).


---

## 🔍 Feature Selection: Filter, Wrapper, Embedded
How we choose the "best" columns from a messy database.
*   **Filter Methods:** Uses statistical scores (Correlations, Chi-Square) to rank features **before** training. (Fast, model-independent).
*   **Wrapper Methods:** Trains the model multiple times with different subsets of features (RFE - Recursive Feature Elimination). (Slow but highly accurate).
*   **Embedded Methods:** The model selects the features **during** training (Lasso L1 penalty, Tree Importance). (The most efficient "Modern" approach).

---

## 👥 Advanced Clustering (DBSCAN)
*   **Density-Based Spatial Clustering (DBSCAN):** Groups points that are "packed tightly" together.
*   **Why it's better than K-Means:** It can find clusters of **any shape** (K-Means only finds circles). Crucially, it automatically identifies **Outliers** as "Noise" instead of forcing them into a cluster.


---

## 7. Multi-Layer Perceptron (Neural Net)
*   **Mechanism:** Layers of interconnected "active" nodes. Uses **Backpropagation** and **Stochastic Gradient Descent (SGD)** to minimize loss.
*   **Loss Function:** 
    *   Binary Classification: **Binary Cross-Entropy (BCE Loss)**
    *   Multi-class Classification: **Categorical Cross-Entropy (CCE Loss)**
    *   Regression: **Mean Squared Error (MSE)**
*   **When to Use:** Large data, complex non-tabular data (Images, Text, Embeddings).

## 🎭 Categorical Encoding
How we transform non-numeric strings into model-ready numbers.

*   **One-Hot Encoding (OHE):** Creates a binary column ($1/0$) for every category. 
    *   *When to Use:* Low-cardinality nominal data (e.g., Red, Blue, Green).
    *   *Gotcha:* **Dimensionality Explosion** if categories >100.
*   **Label / Ordinal Encoding:** Assigns $1, 2, 3...$ to categories.
    *   *When to Use:* Data with a natural rank (e.g., Small, Medium, Large).
    *   *Gotcha:* If used for nominal data (e.g., Apple=1, Banana=2), the model might assume Bananas are "greater than" Apples.
*   **Target Encoding:** Replaces a category with the average target value for that category (e.g., Churn Rate for New York).
    *   *When to Use:* High-cardinality features where OHE is too expensive.
    *   *Gotcha:* Extreme **Target Leakage** risk. Must use Smoothing/K-Folds.
*   **Hashing / Embeddings:** Projects strings into a dense vector (like Two-Tower). 
    *   *When to Use:* Massive, high-cardinality streaming data (e.g., millions of Post IDs).

---

## ⚖️ Imbalanced Data Handling
When you have $99.9\%$ "No Churn" and $0.1\%$ "Churn."

*   **Downsampling:** Removing samples from the "Majority" class to match the "Minority."
    *   *Pros:* Faster training. *Cons:* You throw away useful data.
*   **Upsampling:** Duplicating samples from the "Minority" class.
    *   *Gotcha:* Massive risk of **Overfitting** as the model just memorizes those specific few rows.
*   **SMOTE (Synthetic Minority Over-sampling Technique):** Instead of duplicating rows, it creates **new, synthetic** data points in the space *between* existing minority samples.
    *   *Pros:* Less overfitting than simple duplication.
*   **Weighting:** Many models (XGBoost) allow you to set a `scale_pos_weight`, effectively telling the model that "one churner is worth 100 non-churners." Often preferred over resampling.
*   **Bias-Variance Tradeoff:**
    *   **Bias:** Error from too-simple assumptions (Underfitting).
    *   **Variance:** Error from too-complex patterns (Overfitting).
*   **Hyperparameter:** A "setting" you choose before training (e.g., Learning Rate, Max Depth). 
*   **Parameter:** A "weight" the model learns during training.
