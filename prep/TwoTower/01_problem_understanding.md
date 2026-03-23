# Stage 0: Problem Understanding & Business Goals

Before designing the architecture, an MLE Manager must define the system's "Value."

## 1. The Multi-Sided Marketplace
We do not optimize purely for the User's immediate dopamine. We balance:
1.  **Users:** Receive relevant, high-quality content.
2.  **Creators:** Receive audience reach and incentives to post.
3.  **Advertisers:** Require healthy, attentive users.
4.  **Platform (Meta):** Requires long-term retention (DAU/MAU).

## 2. The Danger of the "Engagement Trap"
*   **The Problem:** If the feed is optimized purely for $P(Click)$ or $P(Like)$, the model will aggressively index on click-bait, controversy, and shallow content. 
*   **The Result:** Initial engagement spikes, followed by **Negative Retention** (users feel the platform is toxic or a waste of time and eventually churn).

## 3. The Solution: Proxy Metrics & The MSI Formula
Because we cannot directly train a real-time model on a 3-month indicator like "Retention," we rely on **Proxy Metrics** via the **MSI (Meaningful Social Interaction)** formula.
We use **Inverse Frequency Weighting** to prioritize high-intent, rare actions over low-intent, common actions.
$$ \text{Value Score} = (1 \times P(\text{Like})) + (3 \times P(\text{Comment})) + (10 \times P(\text{Share})) $$
*Bayesian Optimization is used later in Stage 4 to tune these exact weights against retention A/B tests.*

## 4. Hard System Constraints
A Meta feed must be compiled dynamically the millisecond a user opens the app.
1.  **Scale:** Filter 1 Billion candidate posts.
2.  **Latency Budget:** Return the Top 10 to the viewport in under 200 milliseconds.
3.  **Freshness:** A post created by a close friend 2 minutes ago must be eligible for the very next refresh.
