"""
How a Neuron Works — Complete Python Demonstration
Run: python neuron_demo.py

Walks through every step of a neuron's computation,
then trains a tiny network to solve a classification problem.
"""

import math
import random

# ──────────────────────────────────────────────────
# ACTIVATION FUNCTIONS
# ──────────────────────────────────────────────────

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return max(0.0, z)

def relu_deriv(z):
    return 1.0 if z > 0 else 0.0

def tanh_act(z):
    return math.tanh(z)

def tanh_deriv(z):
    t = math.tanh(z)
    return 1 - t * t

# ──────────────────────────────────────────────────
# SINGLE NEURON — FORWARD PASS
# ──────────────────────────────────────────────────

def neuron_forward(inputs, weights, bias, activation_fn):
    weighted_sum = sum(x * w for x, w in zip(inputs, weights))
    z = weighted_sum + bias
    y = activation_fn(z)
    return y, z, weighted_sum

# ──────────────────────────────────────────────────
# DEMO 1: Step-by-step single neuron
# ──────────────────────────────────────────────────

def demo_single_neuron():
    print("=" * 64)
    print("  DEMO 1: Single Neuron — Step by Step")
    print("=" * 64)

    inputs = [0.5, 0.3, 0.8]
    weights = [0.4, 0.6, 0.2]
    bias = 0.1

    print(f"\n  STEP 1 — Inputs:   x = {inputs}")
    print(f"  STEP 2 — Weights:  w = {weights}")
    print(f"  STEP 3 — Bias:     b = {bias}")

    products = [x * w for x, w in zip(inputs, weights)]
    print(f"\n  STEP 4 — Multiply each input by its weight:")
    for i, (x, w, p) in enumerate(zip(inputs, weights, products)):
        bar = "#" * int(abs(p) * 40)
        sign = "+" if p >= 0 else "-"
        print(f"    x{i+1} * w{i+1} = {x:.2f} * {w:.2f} = {p:.4f}  {sign} {bar}")

    weighted_sum = sum(products)
    print(f"\n  STEP 5 — Weighted sum:  {weighted_sum:.4f}")

    z = weighted_sum + bias
    print(f"  STEP 6 — Add bias:      {weighted_sum:.4f} + {bias:.2f} = {z:.4f}")

    activations = {
        "Sigmoid": sigmoid,
        "ReLU": relu,
        "Tanh": tanh_act,
    }

    print(f"\n  STEP 7 — Apply activation function to z = {z:.4f}:")
    for name, fn in activations.items():
        y = fn(z)
        print(f"    {name:>10}(z) = {y:.6f}")

    print("\n  COMPLETE PIPELINE:")
    print(f"    [{inputs}] -> weights -> sum={weighted_sum:.4f} -> +bias -> z={z:.4f} -> sigmoid -> y={sigmoid(z):.6f}")
    print()

# ──────────────────────────────────────────────────
# DEMO 2: Activation function comparison
# ──────────────────────────────────────────────────

def demo_activation_graphs():
    print("=" * 64)
    print("  DEMO 2: Activation Functions (ASCII Graph)")
    print("=" * 64)

    functions = [
        ("Sigmoid", sigmoid, -6, 6, 0, 1),
        ("ReLU", relu, -3, 3, -0.5, 3),
        ("Tanh", tanh_act, -4, 4, -1, 1),
    ]

    for name, fn, z_min, z_max, y_min, y_max in functions:
        print(f"\n  {name}:")
        rows = 16
        cols = 60
        grid = [[" " for _ in range(cols)] for _ in range(rows)]

        # plot function
        for c in range(cols):
            z = z_min + (z_max - z_min) * c / (cols - 1)
            y = fn(z)
            r = int((1 - (y - y_min) / (y_max - y_min)) * (rows - 1))
            r = max(0, min(rows - 1, r))
            grid[r][c] = "*"

        # axis row
        zero_row = int((1 - (0 - y_min) / (y_max - y_min)) * (rows - 1))
        zero_row = max(0, min(rows - 1, zero_row))

        for r in range(rows):
            label = f"  {y_max - (y_max - y_min) * r / (rows - 1):>6.2f} |"
            line = "".join(grid[r])
            if r == zero_row:
                line = line.replace(" ", "-")
            print(f"  {label}{line}")
        print(f"         {'':>{cols//2}}z=0")
        print()

# ──────────────────────────────────────────────────
# DEMO 3: Training a neuron (gradient descent)
# ──────────────────────────────────────────────────

def demo_training():
    print("=" * 64)
    print("  DEMO 3: Training a Single Neuron")
    print("=" * 64)

    random.seed(42)
    w = [random.uniform(-0.5, 0.5) for _ in range(3)]
    b = random.uniform(-0.5, 0.5)
    lr = 0.5

    inputs = [0.5, 0.3, 0.8]
    target = 1.0

    print(f"\n  Target output: {target}")
    print(f"  Inputs: {inputs}")
    print(f"  Initial weights: [{', '.join(f'{wi:.4f}' for wi in w)}]")
    print(f"  Initial bias: {b:.4f}")
    print(f"  Learning rate: {lr}")
    print()
    print(f"  {'Epoch':>6} | {'Loss':>10} | {'Prediction':>10} | {'w1':>8} {'w2':>8} {'w3':>8} {'bias':>8}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}")

    for epoch in range(1, 201):
        # forward
        z = sum(x * wi for x, wi in zip(inputs, w)) + b
        y = sigmoid(z)

        # loss
        loss = (y - target) ** 2

        # backward
        d_loss_dy = 2 * (y - target)
        dy_dz = sigmoid_deriv(z)
        d_loss_dz = d_loss_dy * dy_dz

        d_loss_dw = [d_loss_dz * x for x in inputs]
        d_loss_db = d_loss_dz

        # update
        w = [wi - lr * dw for wi, dw in zip(w, d_loss_dw)]
        b = b - lr * d_loss_db

        if epoch <= 10 or epoch % 20 == 0 or epoch == 200:
            bar_len = int(y * 30)
            bar = "#" * bar_len + "." * (30 - bar_len)
            print(f"  {epoch:>6} | {loss:>10.6f} | {y:>10.6f} | {w[0]:>8.4f} {w[1]:>8.4f} {w[2]:>8.4f} {b:>8.4f}  |{bar}|")

    print(f"\n  Final prediction: {y:.6f}  (target was {target})")
    print(f"  Final loss:       {loss:.8f}")
    print(f"  Converged!  The neuron learned to output ~{target}.")
    print()

# ──────────────────────────────────────────────────
# DEMO 4: Two-layer network (XOR problem)
# ──────────────────────────────────────────────────

def demo_network():
    print("=" * 64)
    print("  DEMO 4: Two-Layer Network Solving XOR")
    print("=" * 64)

    random.seed(7)

    # XOR truth table
    data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]

    # hidden layer: 2 neurons, each with 2 inputs
    wh = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
    bh = [random.uniform(-1, 1) for _ in range(2)]

    # output layer: 1 neuron with 2 inputs (from hidden)
    wo = [random.uniform(-1, 1) for _ in range(2)]
    bo = random.uniform(-1, 1)

    lr = 1.0

    print(f"\n  XOR Problem: a single neuron CANNOT solve this.")
    print(f"  We need at least 2 layers.\n")
    print(f"  {'Epoch':>6} | {'Loss':>8} | (0,0)->0 | (0,1)->1 | (1,0)->1 | (1,1)->0")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for epoch in range(1, 5001):
        total_loss = 0
        preds = []

        for x, target in data:
            # forward: hidden layer
            zh = [sum(xi * wh[j][i] for i, xi in enumerate(x)) + bh[j] for j in range(2)]
            h = [sigmoid(z) for z in zh]

            # forward: output layer
            zo = sum(h[j] * wo[j] for j in range(2)) + bo
            y = sigmoid(zo)

            loss = (y - target) ** 2
            total_loss += loss
            preds.append(y)

            # backward: output layer
            dl_dy = 2 * (y - target)
            dy_dzo = sigmoid_deriv(zo)
            dl_dzo = dl_dy * dy_dzo

            dl_dwo = [dl_dzo * h[j] for j in range(2)]
            dl_dbo = dl_dzo

            # backward: hidden layer
            dl_dh = [dl_dzo * wo[j] for j in range(2)]
            dl_dzh = [dl_dh[j] * sigmoid_deriv(zh[j]) for j in range(2)]
            dl_dwh = [[dl_dzh[j] * x[i] for i in range(2)] for j in range(2)]
            dl_dbh = [dl_dzh[j] for j in range(2)]

            # update weights
            wo = [wo[j] - lr * dl_dwo[j] for j in range(2)]
            bo = bo - lr * dl_dbo
            wh = [[wh[j][i] - lr * dl_dwh[j][i] for i in range(2)] for j in range(2)]
            bh = [bh[j] - lr * dl_dbh[j] for j in range(2)]

        if epoch <= 5 or epoch % 500 == 0 or epoch == 5000:
            print(f"  {epoch:>6} | {total_loss:>8.5f} | {preds[0]:>8.4f} | {preds[1]:>8.4f} | {preds[2]:>8.4f} | {preds[3]:>8.4f}")

    print(f"\n  After training, the network learned XOR:")
    for x, target in data:
        zh = [sum(xi * wh[j][i] for i, xi in enumerate(x)) + bh[j] for j in range(2)]
        h = [sigmoid(z) for z in zh]
        zo = sum(h[j] * wo[j] for j in range(2)) + bo
        y = sigmoid(zo)
        correct = "yes" if round(y) == target else "NO"
        print(f"    Input {x} -> {y:.4f}  (target {target}, round={round(y)})  {correct}")
    print()


# ──────────────────────────────────────────────────
# RUN ALL DEMOS
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("  HOW A NEURON WORKS — COMPLETE DEMONSTRATION")
    print("  " + "=" * 50)
    print()
    demo_single_neuron()
    demo_activation_graphs()
    demo_training()
    demo_network()
    print("  Done! Open visualizations/neuron_interactive.html")
    print("  in a browser for the interactive version.")
    print()
