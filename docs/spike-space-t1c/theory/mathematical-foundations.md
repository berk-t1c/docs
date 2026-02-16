---
sidebar_position: 4
---

# Mathematical Foundations

This page collects all core equations in one place for reference.

## 1. Membrane Potential Dynamics

### Integrate-and-Fire

$$
V[n+1] = V[n] + I[n]
$$

### Leaky Integrate-and-Fire (subtractive)

$$
V[n+1] = V[n] + I[n] - \lambda
$$

### Leaky Integrate-and-Fire (multiplicative)

$$
V[n+1] = \beta \cdot V[n] + I[n]
$$

## 2. Spike Generation

$$
S(t) = H\bigl(V(t) - \theta\bigr) = \begin{cases} 1 & V(t) \geq \theta \\ 0 & V(t) < \theta \end{cases}
$$

After spiking, the membrane resets: $V(t) \leftarrow 0$.

## 3. Spiking Convolution

At layer $l$, the input current to neuron $(c, h, w)$ is:

$$
I_l[n, c, h, w] = \sum_{c'} \sum_{k_h} \sum_{k_w} W_l[c, c', k_h, k_w] \cdot S_{l-1}[n, c', h + k_h, w + k_w]
$$

where $S_{l-1}$ is the binary spike tensor from the previous layer and $W_l$ are convolutional weights in $[0, 1]$.

## 4. Max Pooling with Index Preservation

$$
y[n, c, h, w],\; \text{idx}[n, c, h, w] = \max_{(k_h, k_w) \in \mathcal{K}} x[n, c, h \cdot s + k_h, w \cdot s + k_w]
$$

Indices are stored for the decoder's unpooling stage.

## 5. STDP Weight Update

For synapse $w_{ij}$ from pre-synaptic neuron $j$ to post-synaptic neuron $i$:

$$
\Delta w_{ij} = \begin{cases}
a^{+} \cdot w_{ij} \cdot (1 - w_{ij}) & \text{if } t_j \leq t_i \quad \text{(LTP)} \\
-a^{-} \cdot w_{ij} \cdot (1 - w_{ij}) & \text{if } t_j > t_i \quad \text{(LTD)}
\end{cases}
$$

## 6. STDP Convergence Metric

$$
C_l = \frac{1}{n_w} \sum_{f} \sum_{i} w_{f,i} (1 - w_{f,i})
$$

Training halts when $C_l < 0.01$.

## 7. Homeostatic Threshold Adaptation

$$
\theta \leftarrow \theta + \theta^{+} \qquad \text{(post-spike)}
$$

$$
\theta \leftarrow \theta - \frac{\theta - \theta_{\text{rest}}}{\tau_\theta} \qquad \text{(decay)}
$$

## 8. Active Spike Hash (ASH)

Given 4D spike activity $(x, y, f, t)$, compress to a binary matrix:

$$
\text{ASH}[f, t] = \begin{cases} 1 & \text{if feature } f \text{ fired at timestep } t \\ 0 & \text{otherwise} \end{cases}
$$

## 9. Jaccard Similarity

For two binary ASH matrices $A$ and $B$:

$$
J(A, B) = \frac{|A \wedge B|}{|A \vee B|}
$$

## 10. SMASH Score

$$
\text{SMASH}(i, j) = J(\text{ASH}_i, \text{ASH}_j) \times \text{IoU}(\text{BBox}_i, \text{BBox}_j)
$$

Instances with $\text{SMASH}(i, j)$ above a threshold are grouped into the same object.

## 11. Evaluation: Informedness

$$
\text{Informedness} = \text{Sensitivity} + \text{Specificity} - 1
$$

$$
= \frac{\text{TP}}{\text{TP} + \text{FN}} + \frac{\text{TN}}{\text{TN} + \text{FP}} - 1
$$

Informedness is the primary metric for the IGARSS 2023 volume-based evaluation (target: 89.1%).
