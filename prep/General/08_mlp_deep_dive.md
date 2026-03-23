# Module 08: MLP Deep Dive (Neural Networks)

An MLP (Multi-Layer Perceptron) is the simplest form of a "Deep" Neural Network. For a Meta MLE role, usually involved in Ranking or Recommendation, understanding how these process features is vital.

***

## 1. The Physical Architecture
*   **The Input Layer:** One neuron for every feature (e.g., User_Age, Post_Timestamp).
*   **The Hidden Layer(s):** Where the "Thinking" happens. Each neuron is a weighted sum of the layer before it, passed through an **Activation Function**.
*   **The Output Layer:** 
    *   **Regression:** 1 neuron (linear output).
    *   **Binary Class:** 1 neuron (Sigmoid output).
    *   **Multi-class:** $N$ neurons (Softmax output).

---

## 2. The Learning Process
### Forward Propagation
Data flows through the network:
1.  **Dot Product:** $Z = W \cdot X + b$ (Weights $\times$ Inputs + Bias).
2.  **Activation:** $A = \sigma(Z)$ (The "Active" signal).

### Backpropagation (The Hard Part)
This is how the model corrects itself using the **Chain Rule**.
1.  Calculate the **Loss** (e.g., BCE Loss) at the output.
2.  Calculate the **Partial Derivative** (Gradient) of the loss with respect to every weight in the network.
3.  **The Goal:** Find how much *each* weight contributed to the final error.

---

## 3. Activation Functions: The Non-Linearity
Without these, an MLP is just a giant linear formula. Activations allow the model to learn complex, curvy patterns.

*   **Sigmoid:** Range $[0, 1]$. Used at the output for binary classification. 
    *   *Gotcha:* **Vanishing Gradient** problem (at extremes, the derivative is zero, meaning the model stops learning).
*   **ReLU (Rectified Linear Unit):** $f(z) = \max(0, z)$.
    *   *Why it's the Standard:* Computationally trivial and avoids vanishing gradients for positive numbers.
    *   *Gotcha:* **Dying ReLU** (If a neuron's weight becomes highly negative, it outputs $0$ forever).
*   **Leaky ReLU:** $f(z) = \max(0.01z, z)$. A fix for the Dying ReLU.

---

## 4. How to Prevent "Deep Overfitting"
Since MLPs have thousands (or millions) of weights, they overfit incredibly fast.

*   **Dropout:** Randomly "killing" a percentage of neurons during training.
    *   *Intuition:* Forces the network to be redundant. No single neuron can "memorize" a pattern.
*   **Early Stopping:** Monitoring the validation loss and stopping training as soon as it starts to rise (even if training loss is still dropping).
*   **Weight Decay (L2):** Penalizing the square of the weights to keep them small.
*   **Batch Normalization:** Normalizing the activations *between* layers. Keeps the internal signal stable, allowing for much faster training (higher learning rates).

---

## 5. Modern Optimizers
While **SGD** is the baseline, real-world DL uses adaptive methods:
*   **Adam (Adaptive Moment Estimation):** It keeps a separate "Learning Rate" for every single weight based on historical gradients (Moments).
    *   *Pros:* Converges much faster than vanilla SGD.
    *   *Cons:* Can sometimes overshoot simple global minima.
