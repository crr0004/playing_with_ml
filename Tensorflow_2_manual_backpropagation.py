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

    def __init__(self):
        super().__init__()

    def run(input, exceptedOutput, layerDefinitions):
        """

        Args:
            input: array of features
            exceptedOutput: array of excepted output for the features
            layerDefinitions: array of number of nureons for the layers
        """
        dtype = tf.float32
        seed=42

        # Create an tensor initializer so we can random initial values
        tensorInitializer = tf.random_uniform_initializer(seed=seed)


        with tf.device("/gpu:0"):
            # Create our input matrices based on the input
            # We define matrices as row major
            inputTensor = tf.constant(input, shape=(len(input), 1), dtype=dtype)
            outputTensor = tf.constant(input, shape=(len(exceptedOutput), 1), dtype=dtype)


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
                layer1Z = tf.matmul(layer1ToLayer2Weights, inputTensor)+layer1Bias
                # Reluu activation
                layer1Activation = tf.math.maximum(layer1Z, 0)

                layer2Z = tf.matmul(layer2ToLayer3Weights, layer1Activation)+layer2Bias
                layer2Activation = tf.math.maximum(layer2Z, 0)
                outputDifference = tf.math.squared_difference(layer2Activation, outputTensor)

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
                layer2BiasDelta = outputCost
                # Equation \frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \delta^l_j
                layer2WeightDelta = tf.matmul(outputCost, layer1Activation, transpose_b=True)

                # This represents the equation 
                # \delta^l_j = ((w^{l+1})^T \delta^{l+1}) \odot \activation'(z^l_j)
                layer1Cost = tf.math.multiply(
                    tf.matmul(layer2ToLayer3Weights, outputCost, transpose_a=True),
                    tf.cast(layer1Activation > 0, dtype)
                    )
                
                layer1BiasDelta = layer1Cost
                layer1WeightDelta = tf.matmul(layer1Cost, inputTensor, transpose_b=True)

                layer2ToLayer3Weights = layer2ToLayer3Weights - layer2WeightDelta
                layer2Bias = layer2Bias - layer2BiasDelta

                layer1ToLayer2Weights = layer1ToLayer2Weights - layer1WeightDelta
                layer1Bias = layer1Bias - layer1BiasDelta



if __name__ == '__main__':
    input = [1, 0, 2, 3]
    exceptedOutput = [1, 1, 1, 1]
    layerDefinitions = [len(input), 10, len(exceptedOutput)]
    Network().run(input, exceptedOutput, layerDefinitions)