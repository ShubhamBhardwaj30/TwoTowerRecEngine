# Module 2: Feature Engineering & Data Prep

When raw behavioral data hits the model, neural networks and trees demand highly constrained numbers, not raw text or gaps. Managing data preparation controls the ultimate statistical variance of your model outputs.

***

## 1. Handling Missing Data (Imputation)
When $20\%$ of a column (like `User_Age`) is `NaN`, Machine Learning libraries like scikit-learn will crash natively. We must handle missingness mathematically:

*   **Row Deletion:** Often dangerous. If users who skipped the "Age" onboarding step churn at a $50\%$ higher rate than normal users, deleting those rows destroys the single most important predictive signal in the dataset. This is called **Missing Not At Random (MNAR)**.
*   **Mean/Median Imputation:** Replacing the `NaN` with the average age of all users. *Con:* It artificially shrinks the statistical variance of the column and assumes the missing user behaves "averagely," which is rarely true.
*   **Stochastic Imputation:** Randomly sampling values from the existing `Age` distribution to fill the gaps. *Pro:* Maintains the natural shape and variance of the data distribution perfectly.
*   **The Shadow Indicator (Best Practice):** Fill `NaN` with an extreme outlier dummy value (e.g., `-1`) AND create a new boolean column `is_age_missing = True`. This allows Tree-based models (like XGBoost) to explicitly recognize that the act of "refusing to provide their age" is a behavioral feature itself.

***

## 2. Handling High-Cardinality Categorical Text
When a text column has thousands of unique strings (e.g., `City`), matrix multiplication cannot compute them. 

*   **One-Hot Encoding (OHE):** Converts each city into a boolean column (0 or 1). *Con:* With 5,000 unique cities, this adds an extra 5,000 extremely sparse columns, triggering the mathematical Curse of Dimensionality and overflowing standard RAM blocks.
*   **Domain-Specific Extraction (Geospatial):** Replacing nominal text strings with structural numbers (e.g., mapping literal "City" to exact numerical Latitude and Longitude variables).
*   **Target Encoding (Mean Encoding):** Replace the text string "New York" with the average target metric of that group—e.g., replacing the string with `0.15` because 15% of users located in New York churned historically. *Pro:* Condenses 5,000 strings into exactly 1 dense numerical column. 
    * *Con (Target Leakage):* It intuitively risks severe data leakage (memorization) by indirectly feeding the $Y$ target variable into the $X$ input feature. 
    * *The Fix:* To execute this safely, ML Engineers use **K-Fold Target Encoding (Out-of-Fold)** or **Leave-One-Out Encoding (LOOE)**. The mean for Row 1 is calculated using *every other row except Row 1*. Furthermore, we apply **Statistical Smoothing** to pull cities with very low sample sizes (e.g. only 2 users) mathematically back toward the global dataset average to avoid tiny-sample variance illusions.
*   **Hashing / Embeddings:** As demonstrated in the Two-Tower module, hashing out-of-vocabulary text into finite buckets, or learning a 64-dimensional dense representation vector of the geometry of the text.

***

## 3. Dimensionality Reduction & PCA
When the number of features ($X$ columns) mathematically approaches the number of observed rows (Users), you fall victim to the **Curse of Dimensionality**. The model gains enough mathematical degrees of freedom to simply memorize the dataset perfectly (Overfitting) rather than learning generalized patterns.

*   **Principal Component Analysis (PCA):** A mathematical projection that condenses thousands of correlated features down into a few dense "Principal Components" while preserving the maximum amount of original statistical **Variance**.
    *   *Pro:* Massively shrinks computation time and defeats the Curse of Dimensionality.
    *   *Con (Loss of Explainability):* The new Principal Components ($PC_1$, $PC_2$) are purely mathematical linear combinations of the original data. They no longer represent independent, real-world concepts. You sacrifice **Interpretability** for execution speed.
*   **Feature Selection (Best Practice):** Blindly throwing all available data through PCA is often a lazy last resort. Advanced ML practitioners prefer aggressive early Feature Selection—actively identifying and keeping only the real-world features that physically influence the Target Variable (using techniques like L1 Regularization to drive bad features to zero, or Tree Feature Importance tracking).
