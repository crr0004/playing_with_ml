import tensorflow as tf
from packaging import version

# Just check we're using tensorflow 2 and throw a warning otherwise
if version.parse(tf.version.VERSION) <= version.parse('2.0.0'):
    raise ImportWarning("Tensorflow version is below 2.0.0")

class Network:
    """
    Implementating backpropagation according to http://neuralnetworksanddeeplearning.com/chap2.html
    with Tensorflow 2.
    """

    def __init__(self, mini_batches=10):
        super().__init__()
        self.mini_batches = mini_batches

    def run(self, input, layerDefinitions):
        """

        Args:
            input: array of features
            exceptedOutput: array of excepted output for the features
            layerDefinitions: array of number of nureons for the layers
        """
        dtype = tf.float32
        seed=42

        # Create an tensor initializer so we can random initial values
        tensorInitializer = tf.random_uniform_initializer(minval=0, maxval=1, seed=seed)


        with tf.device("/gpu:0"):


            # Define these 'back to front' for row major matrices
            layer1ToLayer2Weights = tensorInitializer(
                    (layerDefinitions[1], layerDefinitions[0]), 
                    dtype=dtype
                )
            layer2ToLayer3Weights = tensorInitializer(
                    (layerDefinitions[2], layerDefinitions[1]), 
                    dtype=dtype
                )
            layer1Bias = tensorInitializer(
                    (layerDefinitions[1], 1), 
                    dtype=dtype
                )
            layer2Bias = tensorInitializer(
                    (layerDefinitions[2], 1), 
                    dtype=dtype
                )

            for i in range(10):
                # Strep through each mini_batch out of our input to average out the cost of batch
                batch_range = range(0, 10, self.mini_batches)
                for i in batch_range:
                    # little hacky to get python to multiple assign x and y
                    # going to evaluate the batch
                    mini_batch = input[i : i + batch_range.step]
                    layer2BiasDeltaSum = tf.zeros(layer2Bias.shape, dtype=layer2Bias.dtype)
                    layer2WeightDeltaSum = tf.zeros(layer2ToLayer3Weights.shape, dtype=layer2ToLayer3Weights.dtype)
                    layer1BiasDeltaSum = tf.zeros(layer1Bias.shape, dtype=layer1Bias.dtype)
                    layer1WeightDeltaSum = tf.zeros(layer1ToLayer2Weights.shape, dtype=layer1ToLayer2Weights.dtype)

                    for x, y in mini_batch:
                        inputTensor = tf.constant(x, shape=(layerDefinitions[0], 1), dtype=dtype)
                        # Create our input matrices based on the input
                        # We define matrices as row major
                        # inputTensor, _ = tf.linalg.normalize(inputTensor)
                        outputTensor = tf.constant(y, shape=(layerDefinitions[-1], 1), dtype=dtype)
                        # outputTensor, _ = tf.linalg.normalize(outputTensor)

                        layer1Z = tf.matmul(layer1ToLayer2Weights, inputTensor)+layer1Bias
                        # Reluu activation
                        layer1Activation = tf.math.maximum(layer1Z, 0)

                        layer2Z = tf.matmul(layer2ToLayer3Weights, layer1Activation)+layer2Bias
                        layer2Activation = tf.math.maximum(layer2Z, 0)
                        
                        outputDifference = tf.math.squared_difference(layer2Activation, outputTensor)
                        tf.print(tf.reduce_mean(outputDifference), tf.math.reduce_variance(outputDifference))

                        # Could be called output cost as well. Layer2 is just the final layer here
                        # do an element wise multiplication might be able to do a booleanmask here ¯\_(ツ)_/¯
                        # This represents the equation
                        # \delta^L_j = \frac{\partial C}{\partial a^L_j} \activation'(z^L_j),
                        outputCost = tf.math.multiply(outputDifference, 
                            # This is a lazy way of implementing Reluu differentiation
                            # Basically turn everything to a True or false, with True being > 0,
                            # then turn it all back into numbers.
                            tf.cast(layer2Activation > 0, dtype)
                        )

                        # Equation \frac{\partial C}{\partial b^l_j} = \delta^l_j.
                        layer2BiasDeltaSum += outputCost
                        # Equation \frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \delta^l_j
                        layer2WeightDeltaSum += tf.matmul(outputCost, layer1Activation, transpose_b=True)

                        # This represents the equation 
                        # \delta^l_j = ((w^{l+1})^T \delta^{l+1}) \odot \activation'(z^l_j)
                        layer1Cost = tf.math.multiply(
                            tf.matmul(layer2ToLayer3Weights, outputCost, transpose_a=True),
                            tf.cast(layer1Activation > 0, dtype)
                            )
                    
                        layer1BiasDeltaSum += layer1Cost
                        layer1WeightDeltaSum += tf.matmul(layer1Cost, inputTensor, transpose_b=True)


                    layer2WeightDelta = tf.scalar_mul(1/self.mini_batches, layer2WeightDeltaSum)
                    layer2BiasDelta = tf.scalar_mul(1/self.mini_batches, layer2BiasDeltaSum)

                    layer1WeightDelta = tf.scalar_mul(1/self.mini_batches, layer1WeightDeltaSum)
                    layer1BiasDelta = tf.scalar_mul(1/self.mini_batches, layer1BiasDeltaSum)

                    layer2ToLayer3Weights = layer2ToLayer3Weights - tf.scalar_mul(0.001, layer2WeightDelta)
                    layer2Bias = layer2Bias - tf.scalar_mul(0.001, layer2BiasDelta)

                    layer1ToLayer2Weights = layer1ToLayer2Weights - tf.scalar_mul(0.001, layer1WeightDelta)
                    layer1Bias = layer1Bias - tf.scalar_mul(0.001, layer1BiasDelta)



if __name__ == '__main__':
    import mnist_loader
    import os
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(
        filePath=os.path.dirname(os.path.realpath(__file__)) + "/mnist.pkl.gz"
        )
    training_data = list(training_data)
    # Data is in MNIST format of two arrays, with the second array being the label
    layerDefinitions = [len(training_data[0][0]), 5, len(training_data[0][1])]
    Network().run(training_data, layerDefinitions)