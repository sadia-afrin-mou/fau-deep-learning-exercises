import NeuralNetwork
from Layers import Conv, Pooling, Flatten, FullyConnected, ReLU, SoftMax, Initializers
from Optimization import Optimizers, Loss, Constraints

def build():
    """
    LeNet architecture
    
    returns:
        NeuralNetwork instance
    """
    # optimizer with L2
    optimizer = Optimizers.Adam(learning_rate=5e-4, mu=0.9, rho=0.999)
    regularizer = Constraints.L2_Regularizer(alpha=4e-4)
    optimizer.add_regularizer(regularizer)
    
    # initializers
    weights_initializer = Initializers.He()
    bias_initializer = Initializers.Constant(0.1)
    
    # network
    net = NeuralNetwork.NeuralNetwork(
        optimizer=optimizer,
        weights_initializer=weights_initializer,
        bias_initializer=bias_initializer
    )
    
    # LeNet architecture
    # input: 28x28x1 (MNIST)
    
    # 1st conv layer: 6 filters, 5x5 kernel
    net.append_layer(Conv.Conv(stride_shape=1, convolution_shape=(1, 5, 5), num_kernels=6))
    net.append_layer(ReLU.ReLU())
    
    # 1st pooling layer: 2x2 max pooling
    net.append_layer(Pooling.Pooling(stride_shape=(2, 2), pooling_shape=(2, 2)))
    
    # 2nd conv layer: 16 filters, 5x5 kernel
    net.append_layer(Conv.Conv(stride_shape=1, convolution_shape=(6, 5, 5), num_kernels=16))
    net.append_layer(ReLU.ReLU())
    
    # 2nd pooling layer: 2x2 max pooling
    net.append_layer(Pooling.Pooling(stride_shape=(2, 2), pooling_shape=(2, 2)))
    
    # flatten for fc layers
    net.append_layer(Flatten.Flatten())
    
    # 1st fc layer: 120 neurons
    net.append_layer(FullyConnected.FullyConnected(input_size=16 * 7 * 7, output_size=120))
    net.append_layer(ReLU.ReLU())
    
    # 2nd fc layer: 84 neurons
    net.append_layer(FullyConnected.FullyConnected(input_size=120, output_size=84))
    net.append_layer(ReLU.ReLU())
    
    # output layer: 10 neurons (for 10 digits)
    net.append_layer(FullyConnected.FullyConnected(input_size=84, output_size=10))
    net.append_layer(SoftMax.SoftMax())
    
    # loss layer
    net.loss_layer = Loss.CrossEntropyLoss()
    
    return net
