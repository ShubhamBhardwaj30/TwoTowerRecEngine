# Topic 6: Evaluation Metrics & The Reality Gap

When a Recommendation System shows strong Offline Evaluation metrics (e.g., +5% NDCG, +2% ROC-AUC) but fails during Online A/B Testing (e.g., -2% Session Length), it exposes structural flaws in the business logic rather than the deep learning architecture.

There are four primary causes for this disconnect:

## 1. The Filter Bubble (Exploitation Trap)
When an algorithm perfectly fits historical data (maximizing Offline NDCG), it traps users in a "Filter Bubble." It learns exactly what a user liked yesterday and refuses to show anything else. While this perfectly predicts historical hold-out sets, live humans suffer from **fatigue** and get bored quickly without variety, causing Session Length to plummet.

## 2. Exploration vs Exploitation 
An overly rigid Deep Learning model destroys exploratory and viral content because new posts don't have historical proof of engagement. To fix this, platforms use strategies like **Epsilon-Greedy** or **Upper Confidence Bound (UCB)** algorithms. We forcibly allocate a percentage of the feed to semi-random, unproven content to "explore" the user's evolving tastes and give new creators a chance to surface.

## 3. Goodhart's Law & Metric Misalignment
*"When a measure becomes a target, it ceases to be a good measure."* 
If a Ranking model is solely optimized to predict "Clicks" (a Proxy Metric), it will inevitably learn to surface Click-Bait and Rage-Bait. This artificially spikes the offline Click-Through-Rate but destroys long-term user trust and Retention (the true North Star Metric). This requires the Multi-Task Loss function to deploy heavy negative weights for "quick bounces" (clicking but immediately leaving).

## 4. System Staleness (The Real-Time Lag)
Offline evaluation tests the neural network as if it has perfect, instant access to all user context. In production, if active user interactions (like liking a video 5 seconds ago) aren't pipelined through extremely fast streaming infrastructure (e.g., Apache Kafka / Flink), the Ranker serves outdated Context. The user acts differently in real-time than the model expects.

## 5. Establishing Causality (A/B Testing & Forecasting)
When a metric like Session Length drops during a live rollout, we must mathematically prove that the new model *caused* the drop, rather than external factors (e.g., a holiday, a server outage, or seasonal trends). 
If it is a controlled A/B experiment, we use **Difference-in-Differences (DiD)** to isolate the treatment effect. If evaluating observational time-series data without a clean control group, we use forecasting tools like **Meta Prophet** to predict the counterfactual baseline (what *would* have happened if we didn't launch) and measure the deviation.

## 6. Offline Evaluation: Why We Use NDCG
When measuring the quality of a ranked feed offline, standard metrics like Accuracy or Precision fail because **they don't care about order.** Showing a user their favorite post at position #1 is treated mathematically the same as hiding it at position #50. Recommendation systems require a metric that heavily penalizes bad placement.

**The Math Breakdown:**
1. **CG (Cumulative Gain):** Simply add up the relevance scores (e.g., predict likes) of the posts we showed the user. (Order still doesn't matter here).
2. **D (Discounted):** We aggressively penalize (discount) posts the further down the feed they appear. We divide the relevance score by $log_2(position + 1)$. A highly relevant post placed at position #1 keeps its full value. The exact same post placed at position #10 is mathematically destroyed by the large denominator. This mirrors human scrolling fatigue.
3. **I (Ideal):** We look at the exact same posts, but sort them perfectly (the best posts at the very top). We calculate the DCG of this perfect list. This represents the theoretical maximum score.
4. **N (Normalized):** We divide our model's actual DCG by the Ideal DCG. This yields a final score strictly between `0.0` and `1.0`. Normalization is mandatory because it allows us to average scores across millions of different users, even if User A only had 5 total relevant posts in their entire network and User B had 5,000.

---
### The Formal NDCG Equations

Before calculating, here is the explicit mathematical notation for measuring a list of $p$ items:

**1. Discounted Cumulative Gain (DCG)**  

$DCG_p = \sum_{i=1}^{p} \frac{rel_i}{\log_2(i + 1)}$

*(Where $i$ is the rank order position, and $rel_i$ is the ground-truth relevance score of the item currently sitting at that position).*

**2. Ideal Discounted Cumulative Gain (IDCG)**  

$IDCG_p = \sum_{i=1}^{p} \frac{rel_i^{ideal}}{\log_2(i + 1)}$

*(Where the list has been perfectly sorted descending by relevance).*

**3. Normalized Discounted Cumulative Gain (NDCG)**  

$NDCG_p = \frac{DCG_p}{IDCG_p}$

---
### Step-by-Step NDCG Calculation Example

**The Setup:**
We have 4 candidate posts. We know their *ground-truth relevance* scores based on historical data (3 = highly relevant, 0 = entirely irrelevant). 
- Post A: Relevance 3
- Post B: Relevance 2
- Post C: Relevance 1
- Post D: Relevance 0

**1. The Model's Prediction (DCG)**
Our newly trained model ranks the feed in this order: **[Post B, Post A, Post D, Post C]**.
We calculate the Discounted Cumulative Gain (using base 2 log):
* Position 1 (Post B, rel=2): $2 / \log_2(1+1) = 2 / 1 = 2.0$
* Position 2 (Post A, rel=3): $3 / \log_2(2+1) = 3 / 1.585 = 1.89$
* Position 3 (Post D, rel=0): $0 / \log_2(3+1) = 0 / 2 = 0.0$
* Position 4 (Post C, rel=1): $1 / \log_2(4+1) = 1 / 2.321 = 0.43$
**Total DCG = $2.0 + 1.89 + 0.0 + 0.43 = 4.32$**

**2. The Ideal State (IDCG)**
What is the mathematically perfect way to sort these 4 items? **[Post A, Post B, Post C, Post D]**.
* Position 1 (Post A, rel=3): $3 / \log_2(1+1) = 3 / 1 = 3.0$
* Position 2 (Post B, rel=2): $2 / \log_2(2+1) = 2 / 1.585 = 1.26$
* Position 3 (Post C, rel=1): $1 / \log_2(3+1) = 1 / 2 = 0.5$
* Position 4 (Post D, rel=0): $0 / \log_2(4+1) = 0 / 2.321 = 0.0$
**Total IDCG = $3.0 + 1.26 + 0.5 + 0.0 = 4.76$**

**3. The Final Score (NDCG)**
Since our model scored a 4.32 out of a maximum possible 4.76, its NDCG score is:
**NDCG = $4.32 / 4.76 = 0.907$ (or 90.7%)**
