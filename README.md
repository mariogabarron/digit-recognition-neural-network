# Digit Recognition Neural Network

A neural network implemented from scratch in Python for recognizing digits drawn on a `25x25` pixel grid.

The goal of this project is not to use high-level machine learning frameworks, but to manually implement the internal mathematical mechanisms of a neural network: neurons, weights, biases, forward propagation, cost computation, backpropagation and gradient descent.

> [!NOTE]
> This is a personal educational project built to understand how neural networks work internally, without relying on libraries such as TensorFlow, Keras or PyTorch.

---

## Overview

This project implements a fully connected neural network capable of processing a black-and-white image represented as a `25x25` grid.

Each pixel is treated as an input value between `0` and `1`, producing an input vector of length:

```text
25 x 25 = 625
```

The network outputs 10 activation values, one for each digit from `0` to `9`.

---

## Architecture

The implemented architecture is:

```text
625 → 120 → 60 → 10
```

| Layer          | Size | Description                            |
| -------------- | ---: | -------------------------------------- |
| Input layer    |  625 | One neuron per pixel in the 25x25 grid |
| Hidden layer 1 |  120 | Fully connected hidden layer           |
| Hidden layer 2 |   60 | Fully connected hidden layer           |
| Output layer   |   10 | One neuron per digit, from 0 to 9      |

The prediction is obtained by selecting the output neuron with the highest activation.

```python
prediction = argmax(output_activations)
```

---

## Why This Project

Most neural network projects rely on machine learning frameworks that hide the internal details of training.

This project was built to understand and implement those details manually:

* how a neuron stores its activation, bias and pre-activation value,
* how weight matrices connect consecutive layers,
* how forward propagation computes the output,
* how the cost function measures the error,
* how backpropagation applies the chain rule,
* how gradient descent updates weights and biases.

The main purpose is educational: to connect programming, calculus, linear algebra and optimization in a working neural network.

---

## Core Features

* Neural network implemented from scratch in Python
* Fully connected architecture
* Sigmoid activation function
* Mean squared error cost function
* Manual forward propagation
* Manual backpropagation
* Gradient descent training
* Bias updates
* Digit prediction using output activations
* Tkinter graphical interface for drawing digits
* Custom 25x25 training dataset
* Project report explaining the mathematical foundations

---

## Repository Structure

```text
.
├── estructura_datos.py
├── red_neuronal.py
├── interfaz.py
├── entrenamiento_25x25_correcto.txt
├── requirements.txt
├── README.md
└── Memoria
    └── MemoriaRedNeuronal.pdf
```

### Main files

| File                               | Description                                                                                               |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `estructura_datos.py`              | Defines the neuron structure, sigmoid function and initial weight matrices                                |
| `red_neuronal.py`                  | Implements forward propagation, cost function, backpropagation, gradient descent, training and prediction |
| `interfaz.py`                      | Graphical interface to draw digits and test the neural network                                            |
| `entrenamiento_25x25_correcto.txt` | Training dataset for digits represented as 25x25 grids                                                    |
| `Memoria/MemoriaRedNeuronal.pdf`   | Full project report with mathematical explanations                                                        |

---

## Mathematical Components

### Forward Propagation

For each neuron in a layer, the pre-activation value is computed as:

```text
z = weighted sum + bias
```

Mathematically:

```text
z_k = Σ a_j · w_jk + b_k
```

Then the sigmoid activation function is applied:

```text
a_k = σ(z_k)
```

where:

```text
σ(x) = 1 / (1 + e^(-x))
```

---

### Cost Function

The network uses a mean squared error cost function over the 10 output neurons:

```text
C = (1 / 10) Σ (a_k - y_k)^2
```

where:

* `a_k` is the activation of output neuron `k`,
* `y_k` is the expected value in the one-hot encoded target vector.

---

### Backpropagation

Backpropagation is implemented manually using the chain rule.

The output layer error term is computed as:

```text
δ_k = ((a_k - y_k) / 5) · a_k · (1 - a_k)
```

Then the error is propagated backwards through the hidden layers, allowing the program to compute the gradient of the cost function with respect to every weight.

---

### Gradient Descent

Weights and biases are updated using gradient descent:

```text
parameter = parameter - learning_rate · gradient
```

This gradually reduces the cost and improves the network's predictions.

---

## Graphical Interface

The project includes a simple Tkinter interface that allows the user to:

* draw a digit on a 25x25 grid,
* train the network,
* recognize the drawn digit,
* inspect the output activations,
* clear the canvas,
* save custom examples.

This makes the project interactive and easier to test visually.

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/mariogabarron/digit-recognition-neural-network.git
cd digit-recognition-neural-network
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the interface:

```bash
python interfaz.py
```

---

## Requirements

The project uses:

* Python 3
* NumPy
* Tkinter

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

On some macOS installations using Homebrew, Tkinter may need to be installed separately:

```bash
brew install python-tk
```

---

## Project Report

A full written report is included in the repository:

```text
Memoria/MemoriaRedNeuronal.pdf
```

The report explains the mathematical foundations of the project, including:

* input representation,
* neural network architecture,
* weight matrices,
* sigmoid activation,
* forward propagation,
* cost function,
* chain rule,
* backpropagation,
* gradient descent,
* indexing conventions,
* computational complexity.

---

## Limitations

This project is intentionally simple and educational.

Some limitations are:

* it does not use convolutional layers,
* it is sensitive to how digits are drawn,
* the dataset is small compared to real-world datasets,
* training is slower than optimized machine learning frameworks,
* it is not intended to compete with professional digit-recognition models.

However, these limitations are also what make the project useful for learning: every step of the neural network is visible and implemented manually.

---

## Possible Improvements

Future improvements could include:

* saving and loading trained weights,
* adding a separate test dataset,
* plotting the evolution of the cost function,
* implementing mini-batch training,
* adding regularization,
* experimenting with ReLU activations,
* comparing the implementation with a PyTorch version,
* improving the drawing interface,
* adding more custom training examples.

---

## Purpose

This project was created as a personal learning exercise to understand neural networks from the inside.

Rather than treating a neural network as a black box, the project builds each part manually and shows how calculus, linear algebra and programming come together in a working learning algorithm.
