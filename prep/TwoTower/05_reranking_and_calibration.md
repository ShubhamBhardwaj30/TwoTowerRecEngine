# Stage 4: Re-Ranking (Calibration & Health)

**Goal:** The Ranker creates a perfectly sorted mathematical list. Re-Ranking's job is to destroy that perfection to ensure **Ecosystem Health**.

## 1. Mathematical Scoring vs. Human Experience
If Stage 3 determines you absolute favorite pages are 5 specific Meme accounts, it will rank the top 100 posts as exclusively Meme content.
If you deliver that to the UI, the user is temporarily highly engaged, but ultimately experiences fatigue ("Filter Bubble"). The feed must be manipulated before delivery.

## 2. Multi-Objective Calibration (The Value Formula)
Re-ranking merges the independent predictions of the DLRM.
$$ \text{Feed Score} = (w_1 \cdot P(\text{Like})) + (w_2 \cdot P(\text{Comment})) + (w_3 \cdot P(\text{Share})) $$

*   **Bayesian Optimization (Online):** An MLE Manager does not hard-code these weights. Meta uses BO (BoTorch/Ax) to explore variations in production via A/B testing. BO maps the "Pareto Frontier," finding exact weight combinations that maximize Engagement without cannibalizing the Long-Term Retention metric.

## 3. The Post-Processing Rules
Re-ranking applies explicit final logic to the candidate list:
1.  **Diversity Penalties:** If slots 1 and 2 are from `Author_A`, `Author_A`'s next post is artificially penalized so it drops to slot 8, making room for variety.
2.  **Pacing (Ads):** Ad impressions are deterministically inserted (e.g., every 5th slot) with bid dynamics overlaid on top of user relevance.
3.  **Exploration:** A percentage of the feed is forcefully allocated to "Out of Network" content to help the model learn the bounds of the user's changing interests.

**The Output:** The final 10 perfectly calibrated, diverse candidates are serialized via a GraphQL/Protobuf endpoint payload to the iOS/Android client viewports.
