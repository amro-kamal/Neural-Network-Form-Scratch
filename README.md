## Neural Network From Scratch


This is the code for a simple fully-connected neural network class in numpy. Using this class you can intialize and traind a MLP. The class support all the necessary function to train a neural network, including: 

- [x] forward_propagation(params): To feed the input into the network and get the logits.
- [x] back_propagation(params): Backpropagation with gradient discent.
- [x] gradient_discent(params): Gradient Discent weights update.
- [x] train(params): Function to train the network.
- [x] sigmoid(params), Relu(params), tanh(params), softmax(params): Activation Functions.
- [x] cross_entropy_loss(params), l2_loss(params): Losses.
- [x] predict(params): A function to make inference.
- [x] accuracy(params): For computing the accuracy of the model (for classification).

## Example:
In this example we will train a neural network on mnist dataset to test our neural network class.
### Loading mnist dataset from keras.datasets
Accually we don't need Keras for training the network itself. We will use it just to download mnist dataset to test our code on it.
```python
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

#preprocessing the data (normalization)
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
#minist images are grey scale images of size (28x28), we will reshape them to (28x28)
x_train = x_train.reshape(-1 , 28*28)
x_test = x_test.reshape(-1 , 28*28)
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)

#We need to convert the labels to one-hot vectors, since our neural network class accpects one-hot vectors.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)


```

### Creating and Training the model:
We will define two-layers networks with 128 and 10 neurons (you can choose any number of layer you want).
The neural_net class takes for inputs:
* input:  A batch_size x input_dim numpy arry (1D vector per example).
* labels: as one-hot vectors (numpy array).
* nodes_per_layer: numpy array describing the number of neurons in each layer (for example np.array([128, 64, 10]) for a 3-layers * network).
* activations: The activation function for each layer (for example: np.array(['tanh', 'tanh','sigmoid'])).

```python
from neural_net import neural_net

layers_nums_and_dim = np.array([128,10]) # Two layers neural networks
layers_activations = np.array(['tanh','sigmoid']) #tanh for the first layer and sigmoid for the second layer.
nn = neural_net(x_train , y_train , nodes_per_layer = layers_nums_and_dim, activations = layers_activations )   
#train the network in one line of code
loss , y_pred = nn.train(epochs=1000, lr=0.1, verbose=False)

```

### Prediction for the test data:

```python

y_pred = nn.predict( x_test )
nn.accuracy(y_pred,y_test)

```