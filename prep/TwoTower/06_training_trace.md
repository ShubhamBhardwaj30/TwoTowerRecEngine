# Two-Tower Training: A Step-by-Step Trace

To understand how the **In-Batch Softmax** and **LogQ Correction** work, let's trace a mini-batch with **Batch Size = 2**.

---

### Step 1: Input Features
Suppose we have 2 users and their interacted posts.

| Interaction | User Features ($x$) | Post Features ($y$) | Label |
| :--- | :--- | :--- | :--- |
| **Pair A** | User 1 ($[1, 0]$) | Post 1 ($[0, 1]$) | 1 (Click) |
| **Pair B** | User 2 ($[0, 1]$) | Post 2 ($[1, 0]$) | 1 (Click) |

---

### Step 2: Tower Encoding
The User Tower and Post Tower encode these features into **Embeddings ($D=2$)**.

- **User Embeddings ($U$):** 
  - $U_1 = [0.8, 0.2]$
  - $U_2 = [0.1, 0.9]$
- **Post Embeddings ($P$):** 
  - $P_1 = [0.7, 0.3]$
  - $P_2 = [0.2, 0.8]$

---

### Step 3: Similarity Matrix ($U \cdot P^T$)
The "Matrix Multiplication" $U \cdot P^T$ is just a shorthand for doing **all possible dot-products at once**.

If we have two vectors:
- $U_1 = [a, b]$
- $P_1 = [x, y]$

The **Dot Product** is $ax + by$.

When we do $U \cdot P^T$ for a batch of size 2, the math expands to this:

$$
\begin{pmatrix} U_{1,1} & U_{1,2} \\ U_{2,1} & U_{2,2} \end{pmatrix} 
\times
\begin{pmatrix} P_{1,1} & P_{2,1} \\ P_{1,2} & P_{2,2} \end{pmatrix} 
= 
\begin{pmatrix} (U_1 \cdot P_1) & (U_1 \cdot P_2) \\ (U_2 \cdot P_1) & (U_2 \cdot P_2) \end{pmatrix}
$$

*   **Row 1, Col 1**: Dot product of User 1 and Post 1.
*   **Row 1, Col 2**: Dot product of User 1 and Post 2.

This gives us exactly what we need for the Softmax: **One user's "compatibility score" across all posts in the batch.**

---

### Step 4: Temperature Scaling ($\tau = 0.07$)
We divide all scores by the temperature to "sharpen" the distribution.

$$
\text{Scaled Logits} = \frac{\text{Logits}}{0.07} = \begin{pmatrix} 8.86 & 4.57 \\ 4.86 & 10.57 \end{pmatrix}
$$

---

### Step 5: LogQ Correction (Sampling Bias)
Suppose **Post 1** is very popular (count=100) and **Post 2** is niche (count=1).
- $Q(1) = 100/101 \approx 0.99 \rightarrow \log Q(1) \approx -0.01$
- $Q(2) = 1/101 \approx 0.01 \rightarrow \log Q(2) \approx -4.61$

We subtract $\log Q$ from the columns of the matrix:
$$
\text{Corrected Logits} = \begin{pmatrix} 8.86 - (-0.01) & 4.57 - (-4.61) \\ 4.86 - (-0.01) & 10.57 - (-4.61) \end{pmatrix} 
= \begin{pmatrix} 8.87 & 9.18 \\ 4.87 & 15.18 \end{pmatrix}
$$

*Insight: Post 2 (niche) received a huge boost. This prevents the model from only ever recommending the popular Post 1.*

---

### Step 6: Softmax & Cross-Entropy Loss
The model now treats each row as a classification task.
- **For User 1:** Pick Post 1 or Post 2? (Correct answer: **0** = Post 1)
- **For User 2:** Pick Post 1 or Post 2? (Correct answer: **1** = Post 2)

The **Targets** are $[0, 1]$ (the diagonal).

1.  **Row 0 Probabilities**: $\text{Softmax}([8.87, 9.18]) \approx [0.42, 0.58]$
2.  **Row 1 Probabilities**: $\text{Softmax}([4.87, 15.18]) \approx [0.00, 1.00]$

**Loss:**
The model is very confident about User 2 (high probability for index 1), but is **confused** about User 1 (it actually gave slightly higher probability to Post 2).
The back-propagation will now **push** $U_1$ and $P_1$ closer together and **pull** $U_1$ away from $P_2$.

***

## Summarizing the Shift: Point-wise vs. List-wise

| Feature | Point-wise (BCE Loss) | List-wise (In-Batch Softmax) |
| :--- | :--- | :--- |
| **Perspective** | "Is this user-post pair a 1 or a 0?" | "Which of these 500 posts is the BEST for this user?" |
| **Negative Samples** | Must manually sample 1 or 2 negatives. | Uses every other item in the batch (511+ per user). |
| **Gradient Signal** | Localized to a single pair. | Dense cross-batch information. |
| **Goal** | Individual Classification. | **Relative Ranking.** |

By using the matrix multiplication, you change the model's fundamental logic from *classifying pairs* to *ranking candidates*.

***

### 🚩 Expert Tip: Why all Interaction Labels are Weighted the Same in Retrieval

As you noted, the Two-Tower model is the **"Recall Stage."** 
- **The Goal**: Don't miss *any* relevant post among the 1 Billion candidates.
- **The Logic**: If a user is likely to **Like**, **Comment**, OR **Share**, that post is a "Winner" for Retrieval. It doesn't matter *why* it's a winner at this stage; it just needs to make it into the Top 1,000 for the Ranker to see.
- **The Precision is in Stage 3**: The **DLRMRanker** is where we become hyper-precise. Since it only scores 1,000 items, we can use 3 different "Heads" to calculate exact probabilities for each interaction type.
