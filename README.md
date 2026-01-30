# PicoGrad: The "Tiny" Autograd Engine

## Because size doesn't always matter in ML

PicoGrad is a tiny autograd engine that implements backpropagation (reverse-mode autodiff) over a dynamically built DAG. It's a supercharged version of [micrograd](https://github.com/karpathy/micrograd) with a few extra bells and whistles.

We called it "pico" for the same reason you might call your gaming PC a "little setup" – pure, delightful understatement.

## Features

- Implements a general-purpose Tensor class backed by NumPy
- Distributed computing support, automatic parallelization across CPU cores
- Supports dynamic computational graph construction
- Provides automatic differentiation (autograd) capabilities
- Includes basic neural network building blocks (Neuron, Layer, MLP)
- Offers graph optimization for improved performance
- Graph visualization with Graphviz

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/picograd.git
cd picograd

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from picograd import Value, Tensor

# Create values and perform operations
a = Value(2.0, label='a')
b = Value(3.0, label='b')
c = a * b + a**2
c.backward()

print(f"c = {c.data.data}")  # Forward pass result
print(f"dc/da = {a.grad.data}")  # Gradient of c with respect to a
```

## Neural Networks

PicoGrad can be used to build, train neural networks and visualize the computational graph.

```python
from picograd.nn import MLP, Value

# Create a multi-layer perceptron: 2 inputs, two hidden layers of 8 neurons, 1 output
model = MLP(2, [8, 8, 1])

# Forward pass
x = [Value(1.0), Value(2.0)]
output = model(x)
```

![trained model](./graph/trained_model.png)

## Distributed Computing

PicoGrad supports distributed computing for large tensors, automatically parallelizing operations across multiple CPU cores using Python's multiprocessing.

### Enabling Distributed Mode

```python
from picograd import Tensor, set_distributed_config
import numpy as np

# Enable distributed computing globally
set_distributed_config(
    enabled=True,
    num_workers=4,  # Number of parallel workers (defaults to CPU count)
    min_elements_for_parallel=10000,  # Minimum tensor size to trigger parallelization
    use_processes=True  # Use processes (True) or threads (False)
)

# Create a large tensor - automatically uses distributed operations
large_tensor = Tensor(np.random.randn(1000000, 100))
print(large_tensor)  # Tensor(shape=(1000000, 100), dtype=float32, distributed=True)

# Operations are automatically parallelized
result = large_tensor * 2 + 1
summed = result.sum()
```

### Per-Tensor Control

You can also control distributed computing on a per-tensor basis:

```python
# Force a tensor to use distributed computing
tensor = Tensor(data, distributed=True)

# Disable distributed for a specific tensor
small_tensor = Tensor(data, distributed=False)

# Toggle distributed mode
tensor.enable_distributed()
tensor.disable_distributed()
```

### Supported Operations

Distributed operations include:
- Element-wise: `+`, `-`, `*`, `/`, `**`, negation
- Reductions: `sum()`, `mean()`, `max()`, `min()`
- Utilities: `reshape()`, `transpose()`, indexing

## Graph Optimization

PicoGrad includes graph optimization techniques to improve computational efficiency:

### Initial Graph

![initial graph](./graph/initial_graph.png)

### Optimized Graph

![optimized graph](./graph/optimized_graph.png)

## Running Tests

```bash
python test.py
```
