# NEURALNETWORKS_FROM_SCRATCH

A from-scratch implementation of feedforward neural networks using **NumPy only** — no PyTorch, no TensorFlow, no sklearn. Built to deeply understand the math behind forward propagation, backpropagation, and gradient descent by implementing every piece by hand.

---

## Notebooks

### `neuralnetwork_for_mnist.ipynb`
A 2-layer neural network trained on the classic **MNIST handwritten digit dataset**.

- **Architecture:** 784 → 10 (ReLU) → 10 (Softmax)
- **Dataset:** 60,000 training / 10,000 test images (28×28 grayscale)
- **Task:** 10-class digit classification (0–9)
- **Accuracy:** ~82% on dev set

### `neuralnetwork_for_fashion_mnist.ipynb`
The same architecture adapted for the harder **Fashion-MNIST dataset**.

- **Architecture:** 784 → 10 (ReLU) → 10 (Softmax)
- **Dataset:** 60,000 training / 10,000 test images (28×28 grayscale)
- **Task:** 10-class clothing classification (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Accuracy:** ~74% on dev set

---

## What's Implemented From Scratch

| Component | Details |
|---|---|
| Parameter initialization | Random weights + biases |
| Forward propagation | Linear transform → ReLU → Linear → Softmax |
| Loss | Cross-entropy (implicit via backprop) |
| Backward propagation | Manual chain rule gradients for all layers |
| Parameter update | Vanilla gradient descent |
| Evaluation | Accuracy on held-out dev set |
| Visualization | Per-sample prediction display with true label |

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/AP-HAMESH/NEURALNETWORKS_FROM_SCRATCH.git
cd NEURALNETWORKS_FROM_SCRATCH
```

**2. Install dependencies**
```bash
pip install numpy matplotlib idx2numpy
```

**3. Download the datasets**

- **MNIST:** [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- **Fashion-MNIST:** [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)

Update the `DATA_DIR` path in each notebook to point to where you saved the `.gz` files.

**4. Open and run the notebooks**
```bash
jupyter notebook
```

Run all cells top to bottom. Training prints accuracy every 10 iterations.

---

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- idx2numpy

---

## Results

| Dataset | Dev Accuracy |
|---|---|
| MNIST | ~82% |
| Fashion-MNIST | ~74% |

Fashion-MNIST is intentionally harder — clothing silhouettes (shirt vs. coat vs. pullover) are more ambiguous than digit shapes, which is why it was designed as a drop-in MNIST replacement for benchmarking.

---

## Key Concepts Demonstrated

- **Why ReLU?** Avoids the vanishing gradient problem of sigmoid/tanh for hidden layers.
- **Why Softmax?** Converts raw output scores into a valid probability distribution over classes.
- **One-hot encoding:** Required to compute the gradient of cross-entropy loss cleanly.
- **Dev set split:** 1,000 samples held out from training to evaluate generalization without touching the test set.

---

## License

MIT
