# Notes for book "Practical deeplearning for coders"

## Program vs Model

A program is typically

```
 +--------+       +=========+
 | inputs |------>| program |------> (results)
 +--------+       +=========+
```

A model is:

```
 +--------+       +=========+
 | inputs |------>| model   |------> (results)
 +--------+       +=========+
                     ^
+---------+         /
| weights |--------'
+---------+

```

A model is a **matematical function** that takes inputs and weights. In the case of a neural network, it takes the inputs and multiples for its weights.


```
 +--------+       +=========+
 | inputs |------>| model   |------> (results)----> (loss)
 +--------+       +=========+                         /
                     ^                               /
+---------+         /                               /
| weights |--------´                               /
+---------+                                       /
     ^                                           /
	  \               update                    /
	   ----------------------------------------´
```

### fastai's DataLoaders

To turn our downloaded data into a DataLoaders object we need to tell fastai at least four things:

- What kinds of data we are working with
- How to get the list of items
- How to label these items
- How to create the validation set

## Using volila

```
!pip install voila
!jupyter serverextension enable --sys-prefix voila
```

## Deploying to huggingface-spaces with gradio

1. **Export the model**: This saves the **architecture** and **parameters** to a pickle file. In the case of fastai, also saves the definition to create the `DataLoaders`.

> When we use a model for getting predictions, instead of training, we call it *inference*

2. **Load the Model**
... Complete


## Tensor operations 

- **rank** is the number of axes or **dimensions** in a tensor; shape is the size of each axis of a tensor.
- In pytorch `v.ndim` **is not the number of dimensions** but many times there are confusion
- Numpy arrays and pytorch tensors can finish computations many thousands of times faster than using pure Python.
- pytorch tensors can perform operations such as + - * /
- tensor type: `tensor.type()`

## Gradient "automatic" calculation

In pytorch a tensor can be *marked* to automatically obtain its gradient for subsequent operations you perform.

A key difference between a *metric* and the *loss* function is that the *loss* is to drive automated learning and *metric* is to drive human understanding.

## Sigmoid activation function — Used in binary classifcation

This sigmoid activation is used for **binary problems**, it means it can predict between two categories because it only checks which one has the bigger probability.

## Softmax activation function — Used in multiple category classification

The softmax function is useful when we want to classify wehn we have **more** than two categories

The softmax activation function can be defined in python with:

```python
def softmax(x): 
    return exp(x) / exp(x).sum(dim=1, keepdim=True)
```

Given x is a list, `exp(x)` is the exponential function or `e**x`. How it works?

> Taking the exponential ensures all our numbers are positive, and then dividing by the sum ensures we are going to have a bunch of numbers that add up to 1. The exponential also has a nice property: if one of the numbers in our activations `x` is slightly bigger than the others, the exponential **will amplify** this (since it grows exponentially), which means that in softmax, that number will be closer to 1. So the softmax function *really* wants to pick one class among others.



## Fine tunning basics

A **pretrained model** is a model with some of the parameters already fit. **Fine-tunning** is the process to adjust the already fit parameters to what we need.


