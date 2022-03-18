# Neural-TF-PDE
Neural-TF-PDE is a Python library that solves differential equations using neural networks. It serves as a wrapper for DeepXDE which uses Tensorflow 2.0 as backend.

# Features
1. It solves a variaty of Partial Differential Equations & ODE Systems, such as the Burger Equation, Heat Equation, etc.
1. It can solve the Lorenz ODE System.
1. It is easy & fast to train.
1. It provides fast ODE solvers using a variety of architectures, including PINNs, MsFFN, Multi-Fidelity NNs, ResNet.

# Architectures
1. Physics Informed Neural Networks (PINN) https://arxiv.org/pdf/2110.13361.pdf
1. Multi-Scale Fast-Fourier Networks (MsFFN) https://arxiv.org/pdf/2111.03794.pdf
1. Multi-Fidelity Neural Networks (MFNN) https://www.sciencedirect.com/science/article/pii/S0021999119307260
2. ResNet (https://arxiv.org/abs/1512.03385v1)

Before solving the equations, there are some necessary steps that need to be made, in order to define the problem:

Defining the system of equations.
Defining the data (geometry, boundary conditions).
Selecting a model.
Training the model.

Defining the Equations

```
def pde(x, y):
  eq1 = ...
  eq2 = ...
  return [eq1, eq2]
```

To compute the derivatives of the outputs in your formula, You can use the gradients module, which provides methods for computing the Jacobian & Hessian matrix of a function.

# Jacobian Matrix

The Jacobian matrix can be used to compute the 1st order partial derivatives of a function. The rows of the matrix (0 to N-1) contain
the  terms, while the columns (0 to K-1) contain the
 terms. For example, If you wish to compute the derivative of y1 with respect of x2, then you have to use row = 0, col = 1. If a parameter is None then 0 will be used as a default value.

![Jacobian Matrix](https://github.com/kochlisGit/Neural-TF-PDE/blob/main/Jacobian-matrix-formula.png)

# Hessian Matrix

The hessian matrix can be used to compute the 2nd order partial derivatives of a function. Again You can use the helper function dyy_x with the row & col indicators to grab the partial derivative you desire, according to the hessian matrix above. Additionally, it contains the component parameter, which defines the  term that will be used to compute the partial derivative.

```
def pde(x, y):
  dyy1_x1 = gradients.dyy_x(x, y, 0, 0, 0)
  dyy1_x2 = gradients.dyy_x(x, y, 0, 1, 0)
  dyy2_x2 = gradients.dyy_x(x, y, 0, 1, 1)
  eq = dyy1_x2 - dyy2_x2 + dyy1_x1
  return eq
```

![Hessian Matrix](https://github.com/kochlisGit/Neural-TF-PDE/blob/main/hessian-matrix-formula.png)


**Note: The hessian matrix can also be used to compute higher-order derivatives as well. For example, If you wish the 4th order derivative of a term, You can compute 2 times the hessian matrix. e.g, You can compute the 5th order derivate as:**

```
def pde(x, y):
  dyy_x = gradients.dyy_hessian(x, y)
  dyyyy_x = gradients.dyy_hessian(x, dyyyy_x)
  dyyyyy2_x1 = gradients.dy_x(dyyyy_x, 1, 0, 1)
```

# Defining the Data
Defining the data is a 2-step process:
1. Define the geometry (Interval, Rectangle, Circle, etc.).
1. Define the boundary conditions.

We support the following boundary conditions:
1. DirichletBC: y(x) = func(x)
1. NeumannBC:   y'(x) = func(x)
1. RobinBC:     y'(x) = func(x, y)
1. OperatorBC:  d^n y(x) / dx^n = func(x, y)
1. PointSetBC:  y(x) = constant, if the output of an input is known.
