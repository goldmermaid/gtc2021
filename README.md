#### GTC2021 Tutorial S31692
----

# Dive into Deep Learning:
### Code Side-by-Side with MXNet, PyTorch & TensorFlow

Speaker: Rachel Hu (AWS AI)

---


Deep learning is transforming the world nowadays... 

<center><img src="img/mxnet_pytorch_tf_transp.png" alt="Drawing" style="width: 400px;"/></center>



To fulfill the strong wishes of simpler but more practical deep learning materials, [Dive into Deep Learning](https://d2l.ai/), a unified resource of deep learning was born to achieve the following goals:

- Adopt the 3 most popular deep leraning frameworks: MXNet, PyTorch and TensorFlow;

- Offer depth theory and runnable code, showing readers how to solve problems in practice;

- Allow for rapid updates, both by us, and also by the community at large;

- Be complemented by a forum for interactive discussions of technical details and to answer questions;

- Be freely available for everyone.

## Prerequisites

### GPU Fundamentals

- [Installations with CUDA](https://d2l.ai/chapter_installation/index.html)
- [Basic Operations on GPUs](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/1-use-gpu.ipynb#/)
- [Hardware for deep learning](https://d2l.ai/chapter_computational-performance/hardware.html#gpus-and-other-accelerators)

### Deep Learning Fundamentals

Here are a few concepts that will be the prerequistes for this lecture. Take a look if some of them are not familiar to you! :)

| title                               |  notes    |  slides    |
| ------------------------------ | ---- | ---- |
| Data Manipulation with Ndarray | [D2L Book](https://d2l.ai/chapter_preliminaries/ndarray.html) | [Jupyter Notebook](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/1-ndarray.ipynb#/) |
| Multilayer Perceptron (MLP) | [D2L Book](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html) | [Jupyter Notebook](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/9-mlp-gluon.ipynb#/) |
| Softmax Regression | [D2L Book](https://d2l.ai/chapter_linear-networks/softmax-regression.html) | [Jupyter Notebook](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/7-softmax-regression-gluon.ipynb#/) |
| Fundamental of Convolutional Neural Networks | [D2L Book](https://d2l.ai/chapter_convolutional-neural-networks/index.html) | [Jupyter Notebook](https://nbviewer.jupyter.org/format/slides/github/goldmermaid/gtc2020/blob/master/Notebooks/convolutions.ipynb) |


## Syllabus

In this training, we are applying AlexNet directly on Fashion-MNIST dataset, with 3 frameworks side-by-side.

| Topics |  |  |
| --- | --- | --- |
| AlexNet Overview | [D2L Book](https://d2l.ai/chapter_convolutional-modern/alexnet.html#alexnet) | 
| AlexNet (MXNet) | [Jupyter Notebook](https://nbviewer.jupyter.org/format/slides/github/goldmermaid/gtc2021/blob/master/Notebooks/Alexnet-mxnet.ipynb) | [Nbviewer](notebooks/alexnet-mxnet.slides.html)|
| AlexNet (PyTorch) | [Jupyter Notebook](https://nbviewer.jupyter.org/format/slides/github/goldmermaid/gtc2021/blob/master/alexnet-torch.ipynb) | [Nbviewer](notebooks/alexnet-torch.slides.html)|

## AlexNet

AlexNet, which employed an 8-layer CNN,
won the ImageNet Large Scale Visual Recognition Challenge 2012
by a phenomenally large margin.
This network showed, for the first time,
that the features obtained by learning can transcend manually-designed features, breaking the previous paradigm in computer vision.


The architectures of AlexNet and LeNet are very similar, as we illustrate below.

<center><img src="https://d2l.ai/_images/alexnet.svg" alt="Drawing" style="width: 300px;"/></center>


Note that we provide a slightly streamlined version of AlexNet
removing some of the design quirks that were needed in 2012
to make the model fit on two small GPUs.


The design philosophies of AlexNet and LeNet are very similar,
but there are also significant differences.
First, AlexNet is much deeper than the comparatively small LeNet5.
AlexNet consists of eight layers: five convolutional layers,
two fully-connected hidden layers, and one fully-connected output layer. Second, AlexNet used the ReLU instead of the sigmoid
as its activation function.
Let us delve into the details below.

### Architecture

In AlexNet's first layer, the convolution window shape is $11\times11$.
Since most images in ImageNet are more than ten times higher and wider
than the MNIST images,
objects in ImageNet data tend to occupy more pixels.
Consequently, a larger convolution window is needed to capture the object.
The convolution window shape in the second layer
is reduced to $5\times5$, followed by $3\times3$.
In addition, after the first, second, and fifth convolutional layers,
the network adds maximum pooling layers
with a window shape of $3\times3$ and a stride of 2.
Moreover, AlexNet has ten times more convolution channels than LeNet.

After the last convolutional layer there are two fully-connected layers
with 4096 outputs.
These two huge fully-connected layers produce model parameters of nearly 1 GB.
Due to the limited memory in early GPUs,
the original AlexNet used a dual data stream design,
so that each of their two GPUs could be responsible
for storing and computing only its half of the model.
Fortunately, GPU memory is comparatively abundant now,
so we rarely need to break up models across GPUs these days
(our version of the AlexNet model deviates
from the original paper in this aspect).

### Activation Functions

Besides, AlexNet changed the sigmoid activation function to a simpler ReLU activation function. On one hand, the computation of the ReLU activation function is simpler. For example, it does not have the exponentiation operation found in the sigmoid activation function.
 On the other hand, the ReLU activation function makes model training easier when using different parameter initialization methods. This is because, when the output of the sigmoid activation function is very close to 0 or 1, the gradient of these regions is almost 0, so that backpropagation cannot continue to update some of the model parameters. In contrast, the gradient of the ReLU activation function in the positive interval is always 1. Therefore, if the model parameters are not properly initialized, the sigmoid function may obtain a gradient of almost 0 in the positive interval, so that the model cannot be effectively trained.

### Capacity Control and Preprocessing

AlexNet controls the model complexity of the fully-connected layer
by dropout (:numref:`sec_dropout`),
while LeNet only uses weight decay.
To augment the data even further, the training loop of AlexNet
added a great deal of image augmentation,
such as flipping, clipping, and color changes.
This makes the model more robust and the larger sample size effectively reduces overfitting.
We will discuss data augmentation in greater detail in :numref:`sec_image_augmentation`.


## Coding side-by-side

| Topics |  |
| --- | --- |
| AlexNet Overview | [D2L Book](https://d2l.ai/chapter_convolutional-modern/alexnet.html#alexnet) |
| AlexNet (MXNet) | [Jupyter Notebook](https://nbviewer.jupyter.org/format/slides/github/goldmermaid/gtc2021/blob/master/Notebooks/Alexnet-mxnet.ipynb) |
| AlexNet (PyTorch) | [Jupyter Notebook](https://nbviewer.jupyter.org/format/slides/github/goldmermaid/gtc2021/blob/master/alexnet-torch.ipynb) |

### Bonus Resources

- [AutoGluon](https://autogluon.mxnet.io/) enables easy-to-use and easy-to-extend AutoML with a focus on deep learning and real-world applications spanning image, text, or tabular data;

- [Deep Graph Libray](https://www.dgl.ai/) develops easy-to-use, high performance and scalable Python package for deep learning on graphs;

- [GluonTS](https://gluon-ts.mxnet.io/) supports deep learning based probabilistic time series modeling;

- [TVM](https://tvm.apache.org/): automatic generates and optimizes tensor operators on more backend with better performance for CPUs, GPUs and specialized accelerators.

### Q&A 
If you have any question, please leave us a message at our [discussion forum](https://discuss.d2l.ai/). Have fun diving into deep learning!
