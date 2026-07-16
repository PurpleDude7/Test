# How a Neuron Works in a Neural Network: Complete Guide

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Biological Inspiration](#2-biological-inspiration)
3. [The Artificial Neuron — Step by Step](#3-the-artificial-neuron--step-by-step)
4. [Step 1: Inputs](#step-1-inputs-x)
5. [Step 2: Weights](#step-2-weights-w)
6. [Step 3: Weighted Sum](#step-3-weighted-sum-z)
7. [Step 4: Bias](#step-4-bias-b)
8. [Step 5: Activation Function](#step-5-activation-function-f)
9. [Step 6: Output](#step-6-output-y)
10. [How Neurons Learn — Backpropagation](#4-how-neurons-learn--backpropagation)
11. [Neurons in a Network](#5-neurons-in-a-network)
12. [Complete Numerical Example](#6-complete-numerical-example)
13. [Summary](#7-summary)

---

## 1. The Big Picture

A **neuron** (also called a **node** or **unit**) is the fundamental building block of every neural network. It is a tiny mathematical function that:

1. **Receives** inputs
2. **Processes** them (multiply, sum, transform)
3. **Produces** a single output

Thousands or millions of these neurons, connected together, form the "brain" of an AI model.

```
  ┌─────────────────────────────────────────────────────────┐
  │                    THE NEURON PIPELINE                   │
  │                                                         │
  │   INPUTS      WEIGHTS      SUM       ACTIVATION  OUTPUT │
  │                                                         │
  │   x1 ──┐                                                │
  │         ├──► w1·x1 ──┐                                  │
  │   x2 ──┤             ├──► z = Σ(wi·xi) + b             │
  │         ├──► w2·x2 ──┤              │                   │
  │   x3 ──┘             │              ▼                   │
  │         └──► w3·x3 ──┘        y = f(z) ──────► OUTPUT   │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

---

## 2. Biological Inspiration

The artificial neuron is inspired by the **biological neuron** in the human brain:

```
  BIOLOGICAL NEURON                    ARTIFICIAL NEURON
  ─────────────────                    ─────────────────

  Dendrites (receive signals)    ───►  Inputs (x1, x2, x3...)
  Synapse strength               ───►  Weights (w1, w2, w3...)
  Cell body (processes signals)  ───►  Weighted Sum + Bias (z)
  Axon hillock (fire/not fire)   ───►  Activation Function f(z)
  Axon terminal (sends signal)   ───►  Output (y)
```

| Biological Part | What It Does | Artificial Equivalent |
|---|---|---|
| **Dendrites** | Receive electrical signals from other neurons | Input values (x) |
| **Synapses** | Control how strong each incoming signal is | Weights (w) |
| **Cell Body (Soma)** | Accumulates all incoming signals | Summation (Σ) |
| **Axon Hillock** | Decides whether to fire (threshold) | Activation function |
| **Axon** | Transmits the output signal | Output value (y) |

---

## 3. The Artificial Neuron — Step by Step

Here is the complete mathematical life of a single neuron, from input to output:

```
  ╔═══════════════════════════════════════════════════════════════════╗
  ║                                                                   ║
  ║                    COMPLETE NEURON DIAGRAM                        ║
  ║                                                                   ║
  ║        INPUTS         WEIGHTS        SUMMATION     ACTIVATION     ║
  ║                                                                   ║
  ║    ┌────────┐      ┌─────────┐                                    ║
  ║    │ x1=0.5 │─────►│ w1=0.4  │──┐                                ║
  ║    └────────┘      └─────────┘  │                                 ║
  ║                                 │    ┌──────────┐   ┌──────────┐  ║
  ║    ┌────────┐      ┌─────────┐  ├───►│          │   │          │  ║
  ║    │ x2=0.3 │─────►│ w2=0.6  │──┤    │  z = Σ   │──►│  f(z)    │──► y
  ║    └────────┘      └─────────┘  ├───►│  + bias  │   │          │  ║
  ║                                 │    └──────────┘   └──────────┘  ║
  ║    ┌────────┐      ┌─────────┐  │                                 ║
  ║    │ x3=0.8 │─────►│ w3=0.2  │──┘                                ║
  ║    └────────┘      └─────────┘       ┌──────────┐                 ║
  ║                                      │ bias=0.1 │                 ║
  ║                                      └──────────┘                 ║
  ╚═══════════════════════════════════════════════════════════════════╝
```

---

### Step 1: Inputs (x)

**What**: Raw data fed into the neuron.

**Where they come from**:
- **First layer**: directly from the dataset (pixel values, word embeddings, sensor readings)
- **Hidden layers**: outputs of neurons in the previous layer

```
  Example: Predicting house price

  x1 = 1500    (square footage, normalized)
  x2 = 3       (number of bedrooms)
  x3 = 2       (number of bathrooms)
  x4 = 10      (age of house in years)
```

**Key point**: Inputs are typically **normalized** (scaled to a range like 0-1 or -1 to 1) so that no single input dominates just because of its scale.

```
  RAW DATA                          NORMALIZED DATA
  ──────────                        ────────────────
  x1 = 1500 sq ft    ──normalize──► x1 = 0.75
  x2 = 3 bedrooms    ──normalize──► x2 = 0.50
  x3 = 2 bathrooms   ──normalize──► x3 = 0.40
  x4 = 10 years      ──normalize──► x4 = 0.20
```

---

### Step 2: Weights (w)

**What**: Numbers that control how important each input is to this neuron.

**Analogy**: Think of weights as volume knobs. A high weight means "this input matters a lot." A low (or negative) weight means "this input matters little (or has an inverse effect)."

```
  ┌──────────────────────────────────────────────────────────┐
  │              WEIGHT AS IMPORTANCE                         │
  │                                                          │
  │   Input: sq footage  ──►  w = 0.9  ──►  VERY IMPORTANT  │
  │   Input: bedrooms    ──►  w = 0.5  ──►  MODERATE        │
  │   Input: bathrooms   ──►  w = 0.3  ──►  LESS IMPORTANT  │
  │   Input: age         ──►  w = -0.4 ──►  NEGATIVE EFFECT │
  │                                                          │
  │   Visualization of weight magnitudes:                    │
  │                                                          │
  │   w1 = 0.9  ████████████████████░░░░  (strong positive)  │
  │   w2 = 0.5  ███████████░░░░░░░░░░░░  (moderate)         │
  │   w3 = 0.3  ██████░░░░░░░░░░░░░░░░░  (weak positive)   │
  │   w4 = -0.4 ▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░  (negative)        │
  └──────────────────────────────────────────────────────────┘
```

**How they start**: Weights are initialized **randomly** (small values near zero) before training begins.

**How they change**: During training, weights are adjusted through **backpropagation** (explained later).

---

### Step 3: Weighted Sum (z)

**What**: Multiply each input by its weight, then add everything up.

**Formula**:

```
  z = (x1 * w1) + (x2 * w2) + (x3 * w3) + ... + (xn * wn) + b

  In mathematical notation:
         n
  z  =  Σ  (xi * wi) + b
        i=1
```

**Concrete Example**:

```
  Inputs:    x1=0.5,  x2=0.3,  x3=0.8
  Weights:   w1=0.4,  w2=0.6,  w3=0.2
  Bias:      b=0.1

  Step-by-step calculation:
  ─────────────────────────

  Multiply each input by its weight:

      x1 * w1 = 0.5 * 0.4 = 0.20
      x2 * w2 = 0.3 * 0.6 = 0.18
      x3 * w3 = 0.8 * 0.2 = 0.16
                             ─────
  Sum all products:          0.54

  Add bias:             0.54 + 0.1 = 0.64

  z = 0.64
```

**Vector notation** (how it is computed efficiently in code):

```
  x = [0.5, 0.3, 0.8]       (input vector)
  w = [0.4, 0.6, 0.2]       (weight vector)

  z = dot(x, w) + b
  z = (x . w) + b
  z = 0.54 + 0.1
  z = 0.64
```

---

### Step 4: Bias (b)

**What**: An extra learnable number added to the weighted sum. It shifts the activation function left or right.

**Why it matters**: Without bias, the neuron can only learn functions that pass through the origin (0,0). Bias gives the neuron freedom to fit data that doesn't go through zero.

```
  ┌──────────────────────────────────────────────────────────┐
  │           WITHOUT BIAS vs WITH BIAS                       │
  │                                                          │
  │   Without Bias (b=0):          With Bias (b=2):          │
  │                                                          │
  │   y│                           y│                        │
  │    │      /                     │          /             │
  │    │    /                       │        /               │
  │    │  /                         │      /                 │
  │    │/                           │    /                   │
  │   ─┼──────── x                  │  /                     │
  │    │                            │/                       │
  │    │                           ─┼──────── x              │
  │                                 │                        │
  │   Line MUST go through          Line can be SHIFTED      │
  │   the origin (0,0)              up/down/left/right       │
  └──────────────────────────────────────────────────────────┘
```

**Analogy**: The bias is like the **y-intercept** in the equation `y = mx + b`. Without it, every line must go through the origin.

---

### Step 5: Activation Function (f)

**What**: A mathematical function applied to the weighted sum `z` to introduce **non-linearity**.

**Why it is critical**: Without activation functions, stacking layers would just produce another linear function. Non-linearity lets the network learn complex patterns like curves, edges, and abstract concepts.

```
  ┌───────────────────────────────────────────────────────────┐
  │    WHY WE NEED NON-LINEARITY                              │
  │                                                           │
  │    Linear only:              With activation function:    │
  │                                                           │
  │    y│                        y│        ___                │
  │     │      /                  │      /    \               │
  │     │    /                    │    /       \              │
  │     │  /                      │  /         \             │
  │     │/                        │/            \            │
  │    ─┼──────── x              ─┼──────────────\── x       │
  │     │                         │               \          │
  │                                                          │
  │    Can only learn             Can learn ANY              │
  │    straight lines             complex pattern!           │
  └───────────────────────────────────────────────────────────┘
```

#### The Major Activation Functions

##### 1. Sigmoid (Logistic)

```
  Formula:  f(z) = 1 / (1 + e^(-z))
  Range:    (0, 1)
  Use:      Output layer for binary classification

  Graph:
        1.0 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
             │                              _______________
             │                         ___/
        0.5  │                      __/
             │                   __/
             │               ___/
        0.0 ─│──────────────/─────────────────────────────
            -6  -5  -4  -3  -2  -1   0   1   2   3   4   5   6

  Properties:
    - Squashes ANY number to range (0, 1)
    - Smooth, differentiable everywhere
    - Problem: "vanishing gradient" for very large/small z
    - Interpretation: probability (e.g., 80% chance it's a cat)
```

##### 2. Tanh (Hyperbolic Tangent)

```
  Formula:  f(z) = (e^z - e^(-z)) / (e^z + e^(-z))
  Range:    (-1, 1)
  Use:      Hidden layers (centers data around zero)

  Graph:
        1.0 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
             │                              _______________
             │                         ___/
        0.0  │─────────────────────__/────────────────────
             │                ___/
             │           ____/
       -1.0 ─│__________/────────────────────────────────
            -6  -5  -4  -3  -2  -1   0   1   2   3   4   5   6

  Properties:
    - Zero-centered output (better for hidden layers)
    - Stronger gradients than sigmoid
    - Still suffers from vanishing gradients at extremes
```

##### 3. ReLU (Rectified Linear Unit) — MOST POPULAR

```
  Formula:  f(z) = max(0, z)
  Range:    [0, infinity)
  Use:      Default choice for hidden layers in most networks

  Graph:
             │                          /
             │                        /
             │                      /
             │                    /
             │                  /
        0.0  │________________/
             │
            ─┼──────────────────────────────────────────
            -6  -5  -4  -3  -2  -1   0   1   2   3   4   5   6

  Properties:
    - Dead simple: output z if positive, else 0
    - Computationally very fast
    - Solves vanishing gradient problem
    - Problem: "dying ReLU" — neurons can permanently die
      (output 0 for all inputs, never recover)
```

##### 4. Leaky ReLU

```
  Formula:  f(z) = z if z > 0, else 0.01 * z
  Range:    (-infinity, infinity)
  Use:      When dying ReLU is a problem

  Graph:
             │                          /
             │                        /
             │                      /
             │                    /
             │                  /
        0.0  │_______________./
             │             .
            ─┼──────────.───────────────────────────────
            -6  -5  -4  -3  -2  -1   0   1   2   3   4   5   6

    (small negative slope instead of flat zero)

  Properties:
    - Allows small gradient for negative values
    - Prevents dying neurons
    - 0.01 is the standard leak coefficient (can be tuned)
```

##### 5. Softmax (Special — Multi-class Output)

```
  Formula:  f(zi) = e^zi / Σ(e^zj)  for each class i
  Range:    (0, 1) and all outputs sum to 1.0
  Use:      Output layer for multi-class classification

  Example:
    Raw scores (logits):    z = [2.0,  1.0,  0.5]
    After softmax:          y = [0.59, 0.24, 0.17]
                                 ─────────────────
                                 Sum = 1.00

    Interpretation:
      Class A: 59% probability
      Class B: 24% probability
      Class C: 17% probability
```

#### Activation Function Comparison

```
  ┌───────────────┬──────────┬────────────────┬──────────────────────┐
  │  Function     │  Range   │  Best For      │  Key Issue           │
  ├───────────────┼──────────┼────────────────┼──────────────────────┤
  │  Sigmoid      │ (0, 1)   │ Binary output  │ Vanishing gradient   │
  │  Tanh         │ (-1, 1)  │ Hidden layers  │ Vanishing gradient   │
  │  ReLU         │ [0, inf) │ Hidden layers  │ Dying neurons        │
  │  Leaky ReLU   │ (-inf,inf│ Hidden layers  │ Extra hyperparameter │
  │  Softmax      │ (0, 1)   │ Multi-class    │ Expensive to compute │
  └───────────────┴──────────┴────────────────┴──────────────────────┘
```

---

### Step 6: Output (y)

**What**: The final value the neuron produces after the activation function is applied.

```
  y = f(z)

  In our running example:
    z = 0.64
    Using sigmoid:  y = 1 / (1 + e^(-0.64))
                    y = 1 / (1 + 0.527)
                    y = 1 / 1.527
                    y = 0.655

    Using ReLU:     y = max(0, 0.64)
                    y = 0.64

    Using tanh:     y = tanh(0.64)
                    y = 0.565
```

**What happens to this output?**

```
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │   If this neuron is in a HIDDEN LAYER:                  │
  │   ──────────────────────────────────                    │
  │   The output (y=0.655) becomes an INPUT                 │
  │   to neurons in the NEXT layer                          │
  │                                                         │
  │         [This Neuron]──► y=0.655 ──► [Next Neuron 1]   │
  │                                 ──► [Next Neuron 2]    │
  │                                 ──► [Next Neuron 3]    │
  │                                                         │
  │   If this neuron is in the OUTPUT LAYER:                │
  │   ─────────────────────────────────                     │
  │   The output IS the prediction                          │
  │                                                         │
  │         [This Neuron]──► y=0.655 ──► "65.5% chance     │
  │                                       this is a cat"    │
  └─────────────────────────────────────────────────────────┘
```

---

## 4. How Neurons Learn — Backpropagation

The magic of neural networks is that neurons **learn** the right weights and biases through training. Here is the complete learning cycle:

### The Training Loop (Forward Pass + Backward Pass)

```
  ╔══════════════════════════════════════════════════════════════════╗
  ║                    THE COMPLETE TRAINING LOOP                    ║
  ║                                                                  ║
  ║   ┌─────────────────── FORWARD PASS ───────────────────────┐     ║
  ║   │                                                         │     ║
  ║   │  Input ──► Weights ──► Sum ──► Activation ──► Output   │     ║
  ║   │  (x)       (w)         (z)     f(z)           (y)      │     ║
  ║   └─────────────────────────────────────────────────────────┘     ║
  ║                                          │                        ║
  ║                                          ▼                        ║
  ║                                   ┌──────────────┐               ║
  ║                                   │ LOSS FUNCTION │               ║
  ║                                   │ Compare y to  │               ║
  ║                                   │ true label    │               ║
  ║                                   │ L = (y-t)^2   │               ║
  ║                                   └──────┬───────┘               ║
  ║                                          │                        ║
  ║   ┌─────────────────── BACKWARD PASS ──────────────────────┐     ║
  ║   │                                          │              │     ║
  ║   │  Update ◄── Gradients ◄── Chain Rule ◄──┘              │     ║
  ║   │  weights    (∂L/∂w)       (calculus)                    │     ║
  ║   │                                                         │     ║
  ║   │  w_new = w_old - learning_rate * (∂L/∂w)               │     ║
  ║   └─────────────────────────────────────────────────────────┘     ║
  ║                                          │                        ║
  ║                                          ▼                        ║
  ║                               REPEAT for many epochs              ║
  ╚══════════════════════════════════════════════════════════════════╝
```

### Step-by-Step Learning

#### Phase 1: Forward Pass (make a prediction)

```
  Input: x = [0.5, 0.3, 0.8]
  Initial random weights: w = [0.4, 0.6, 0.2]
  Bias: b = 0.1
  True answer: t = 1.0 (target)

  z = (0.5)(0.4) + (0.3)(0.6) + (0.8)(0.2) + 0.1 = 0.64
  y = sigmoid(0.64) = 0.655

  Prediction: 0.655     (but the correct answer is 1.0!)
```

#### Phase 2: Calculate the Error (Loss)

```
  Using Mean Squared Error (MSE):

  Loss = (y - t)^2
       = (0.655 - 1.0)^2
       = (-0.345)^2
       = 0.119

  The error is 0.119 — the neuron's prediction was OFF.
```

#### Phase 3: Backward Pass (Backpropagation)

Calculate how much each weight contributed to the error using the **chain rule** from calculus:

```
  The chain rule lets us find: "How does changing w1 affect the loss?"

  ∂Loss/∂w1 = ∂Loss/∂y  *  ∂y/∂z  *  ∂z/∂w1
               ─────────    ──────     ───────
               How loss     How        How z changes
               changes      activation when w1
               with y       changes    changes
                            with z

  Computing each piece:

  ∂Loss/∂y = 2(y - t) = 2(0.655 - 1.0) = -0.690

  ∂y/∂z = sigmoid(z) * (1 - sigmoid(z))
        = 0.655 * (1 - 0.655)
        = 0.655 * 0.345
        = 0.226

  ∂z/∂w1 = x1 = 0.5

  Full gradient for w1:
  ∂Loss/∂w1 = -0.690 * 0.226 * 0.5 = -0.078
```

#### Phase 4: Update the Weights

```
  Using gradient descent with learning_rate = 0.1:

  w1_new = w1_old - learning_rate * gradient
         = 0.4   - 0.1 * (-0.078)
         = 0.4   + 0.0078
         = 0.4078

  The weight INCREASED slightly, pushing the output
  closer to the target of 1.0.

  Similarly for all weights:
  ┌──────────┬──────────┬───────────┬──────────┐
  │  Weight  │   Old    │ Gradient  │   New    │
  ├──────────┼──────────┼───────────┼──────────┤
  │  w1      │  0.4000  │  -0.0780  │  0.4078  │
  │  w2      │  0.6000  │  -0.0468  │  0.6047  │
  │  w3      │  0.2000  │  -0.1248  │  0.2125  │
  │  bias    │  0.1000  │  -0.1559  │  0.1156  │
  └──────────┴──────────┴───────────┴──────────┘

  After update, the neuron now computes:
  z = (0.5)(0.4078) + (0.3)(0.6047) + (0.8)(0.2125) + 0.1156
  z = 0.6553
  y = sigmoid(0.6553) = 0.658

  Old prediction: 0.655 → New prediction: 0.658
  Tiny improvement! Repeat thousands of times to converge.
```

### Gradient Descent Visualized

```
  Loss
   │
   │\
   │ \
   │  \
   │   \        ← Start here (random weights, high loss)
   │    \
   │     \
   │      \     ← Gradient points downhill
   │       \
   │        \.
   │         '·.
   │            '·._
   │                '·.__        ← Each step reduces loss
   │                     '·.___
   │                           ──── ← Minimum (best weights!)
   └─────────────────────────────── weight value

   Learning rate too LARGE:        Learning rate too SMALL:

   Loss                            Loss
    │\                               │\
    │ \    /\                         │ \
    │  \  /  \   ← Overshoots!       │  \
    │   \/    \  /                    │   ·
    │          \/                     │    ·    ← Crawls slowly
    │                                │     ·
    └──────────────                  │      ·.....
                                     └────────────
```

---

## 5. Neurons in a Network

A single neuron is limited. The real power comes from connecting many neurons in **layers**:

```
  ┌────────────────────────────────────────────────────────────────┐
  │                                                                │
  │    INPUT LAYER      HIDDEN LAYER 1    HIDDEN LAYER 2    OUTPUT │
  │    (features)       (patterns)        (combinations)    LAYER  │
  │                                                                │
  │    ┌───┐            ┌───┐             ┌───┐                    │
  │    │ x1├────────────┤ h1├─────────────┤ h4├──┐                 │
  │    └─┬─┘    ╲       └─┬─┘     ╲       └─┬─┘  │   ┌───┐        │
  │      │       ╲        │        ╲         │    ├───┤ o1│──► y   │
  │    ┌─┴─┐      ╲     ┌─┴─┐      ╲      ┌─┴─┐  │   └───┘        │
  │    │ x2├───────╲────┤ h2├──────╲──────┤ h5├──┘                 │
  │    └─┬─┘        ╲   └─┬─┘       ╲     └─┬─┘                   │
  │      │           ╲    │          ╲       │                     │
  │    ┌─┴─┐          ╲ ┌─┴─┐        ╲    ┌─┴─┐                   │
  │    │ x3├───────────╲┤ h3├─────────╲───┤ h6│                   │
  │    └───┘             └───┘             └───┘                   │
  │                                                                │
  │    3 neurons         3 neurons         3 neurons    1 neuron   │
  │                                                                │
  │    Each connection has its own weight (w)                       │
  │    Total weights: (3×3) + (3×3) + (3×1) = 21 weights          │
  │    Total biases:  3 + 3 + 1 = 7 biases                        │
  │    Total learnable parameters: 28                              │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

### What Each Layer Learns

```
  ┌────────────────────────────────────────────────────────────────┐
  │  Example: Image Recognition (What does a "cat" look like?)     │
  │                                                                │
  │  INPUT         LAYER 1          LAYER 2          OUTPUT        │
  │  (pixels)      (edges)          (shapes)         (objects)     │
  │                                                                │
  │  ░░██░░        ─── (lines)      ◯ (circles)     🐱 Cat: 95%  │
  │  ░████░        │ (verticals)    △ (triangles)    🐶 Dog: 3%   │
  │  ██████        / (diagonals)    □ (rectangles)   🐦 Bird: 2%  │
  │  ░████░        \ (curves)       ◇ (patterns)                  │
  │  ░░██░░                                                        │
  │                                                                │
  │  Raw pixels    Simple           Complex          Final         │
  │  (numbers)     features         features         decision      │
  └────────────────────────────────────────────────────────────────┘
```

---

## 6. Complete Numerical Example

Let's trace a complete forward pass through a tiny 2-layer network:

### Problem: Is this point above or below a line?

```
  Input: (x1, x2) = (0.6, 0.4)
  Target: y = 1 (above the line)

  Network architecture:
    Input layer:  2 neurons (x1, x2)
    Hidden layer: 2 neurons (h1, h2) with ReLU
    Output layer: 1 neuron  (o1) with Sigmoid
```

### Forward Pass

```
  ════════════════════════════════════════════════════
  LAYER 1: Input → Hidden (2 neurons, ReLU)
  ════════════════════════════════════════════════════

  Neuron h1:
    Weights: w11=0.3, w12=0.5    Bias: b1=0.1

    z1 = (0.6 × 0.3) + (0.4 × 0.5) + 0.1
       = 0.18 + 0.20 + 0.1
       = 0.48

    h1 = ReLU(0.48) = max(0, 0.48) = 0.48   ✓ positive, passes through

  Neuron h2:
    Weights: w21=0.7, w22=-0.4   Bias: b2=-0.2

    z2 = (0.6 × 0.7) + (0.4 × -0.4) + (-0.2)
       = 0.42 + (-0.16) + (-0.2)
       = 0.06

    h2 = ReLU(0.06) = max(0, 0.06) = 0.06   ✓ positive, passes through

  ════════════════════════════════════════════════════
  LAYER 2: Hidden → Output (1 neuron, Sigmoid)
  ════════════════════════════════════════════════════

  Neuron o1:
    Weights: w31=0.8, w32=0.6    Bias: b3=-0.3

    z3 = (0.48 × 0.8) + (0.06 × 0.6) + (-0.3)
       = 0.384 + 0.036 + (-0.3)
       = 0.12

    o1 = sigmoid(0.12) = 1/(1 + e^(-0.12))
       = 1/(1 + 0.887)
       = 1/1.887
       = 0.530

  ════════════════════════════════════════════════════
  RESULT
  ════════════════════════════════════════════════════

  Prediction:  0.530  (53% confident it is above the line)
  Target:      1.000  (it IS above the line)
  Loss (MSE):  (0.530 - 1.0)^2 = 0.221

  The network needs more training!
  After hundreds of iterations, it will learn the correct
  weights to classify points accurately.
```

### Training Convergence

```
  Epoch 1:      Loss = 0.2210    Prediction = 0.530
  Epoch 10:     Loss = 0.1450    Prediction = 0.619
  Epoch 50:     Loss = 0.0520    Prediction = 0.772
  Epoch 100:    Loss = 0.0180    Prediction = 0.866
  Epoch 200:    Loss = 0.0045    Prediction = 0.933
  Epoch 500:    Loss = 0.0004    Prediction = 0.980
  Epoch 1000:   Loss = 0.0001    Prediction = 0.993   ← converged!

  Loss over time:
  0.25│*
      │ *
  0.20│  *
      │   *
  0.15│    **
      │      **
  0.10│        ***
      │           ****
  0.05│               *****
      │                    ********
  0.00│                            **********************
      └─────────────────────────────────────────────────── Epochs
       0    100   200   300   400   500   600   700   800  1000
```

---

## 7. Summary

### The Complete Neuron Pipeline (One Final View)

```
  ╔═══════════════════════════════════════════════════════════════════╗
  ║                                                                   ║
  ║  1. INPUTS         Raw data or outputs from previous layer        ║
  ║     x1, x2, ..xn  ──────────────────────┐                        ║
  ║                                          │                        ║
  ║  2. WEIGHTS        Learnable importance  │                        ║
  ║     w1, w2, ..wn   of each input         │                        ║
  ║         │                                │                        ║
  ║         ▼                                ▼                        ║
  ║  3. WEIGHTED SUM   z = Σ(wi × xi) + b                            ║
  ║         │           Dot product + bias                            ║
  ║         ▼                                                         ║
  ║  4. BIAS (b)       Shifts the decision boundary                   ║
  ║         │           Added to weighted sum                         ║
  ║         ▼                                                         ║
  ║  5. ACTIVATION     f(z) = sigmoid / ReLU / tanh / etc.           ║
  ║         │           Introduces non-linearity                      ║
  ║         ▼                                                         ║
  ║  6. OUTPUT (y)     Single number sent to next layer               ║
  ║                     or used as final prediction                   ║
  ║                                                                   ║
  ║  7. LEARNING       Backpropagation adjusts weights                ║
  ║                     using gradients to minimize loss              ║
  ║                                                                   ║
  ╚═══════════════════════════════════════════════════════════════════╝
```

### Key Takeaways

| Concept | One-Line Summary |
|---|---|
| **Neuron** | A function: y = f(w . x + b) |
| **Weights** | Control input importance; learned during training |
| **Bias** | Shifts the decision boundary; learned during training |
| **Activation** | Adds non-linearity so networks can learn complex patterns |
| **Forward Pass** | Input → weighted sum → activation → output |
| **Loss Function** | Measures how wrong the prediction is |
| **Backpropagation** | Calculates how to adjust each weight to reduce loss |
| **Gradient Descent** | The algorithm that actually updates the weights |
| **Learning Rate** | Controls how big each weight update step is |
| **Epoch** | One complete pass through the entire training dataset |

---

*See `visualizations/neuron_interactive.html` for an interactive visualization of these concepts.*
