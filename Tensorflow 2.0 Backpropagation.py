import tensorflow as tf
from packaging import version

# Just check we're using tensorflow 2 and throw a warning otherwise
if version.parse(tf.version.VERSION) <= version.parse('2.0.0'):
    raise ImportWarning("Tensorflow version is below 2.0.0")

def run(input, exceptedOutput, layerDefinitions):
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
        layer2Bias = tensorInitializer(
                (layerDefinitions[1], 1), 
                dtype=dtype
            )
        layer3Bias = tensorInitializer(
                (layerDefinitions[2], 1), 
                dtype=dtype
            )

        layer1Z = tf.matmul(layer1ToLayer2Weights, inputTensor)+layer2Bias
        # Reluu activation
        layer1Activation = tf.math.maximum(layer1Z, 0)

        layer2Z = tf.matmul(layer2ToLayer3Weights, layer1Activation)+layer3Bias
        layer2Activation = tf.math.maximum(layer2Z, 0)
        outputDifference = tf.math.squared_difference(layer2Activation, outputTensor)




        # matrix2 = tf.Variable(tf.ones((n, n), dtype=dtype))
        # product = tf.matmul(matrix1, matrix2)
        # print(product)



if __name__ == '__main__':
    input = [1, 0, 2, 3]
    exceptedOutput = [1, 1, 1, 1]
    layerDefinitions = [len(input), 10, len(exceptedOutput)]
    run(input, exceptedOutput, layerDefinitions)