import tensorflow as tf
import gzip
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 28
DATA_DIR = '~/mnist/data'
TEST_DIR = 'test'
TRAIN_DIR = 'train'
PIXEL_DEPTH = 255.
NUM_CHANNELS = 1
NUM_LABELS = 10

def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
  return data

def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

  print(labels)
  return labels


# determines classification loss using cross-entropy loss
def classification_loss(Y, Y_hat, layer_weights, weight_decay):
    loss_D = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_hat))/tf.to_float(tf.shape(Y)[0])
    # add weight decay 
    loss_W = 0
    for i in layer_weights:
        loss_W += (weight_decay/2) * tf.square(tf.norm(i))

    loss = loss_D + loss_W
    return loss

def classification_accuracy(Y, Y_hat):
    # calculate classification accuracy
    threshold = tf.cast(tf.argmax(Y_hat, axis=1), tf.float32)
    outputs = tf.cast(tf.argmax(Y, axis=1), tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(threshold, outputs), tf.float32))

    return accuracy

conv_layer = 0
def init_conv_layers(filter_sizes):
    global conv_layer
    conv_layers = []
    for i in filter_sizes:
        conv_layers.append(tf.get_variable("conv_" + str(conv_layer), shape=i, initializer=tf.contrib.layers.xavier_initializer()))
        conv_layer += 1

    return conv_layers

fc_layer = 0
def init_fc_layers(input_layer, layer_sizes):
    """ Initializes the FC layers, returning the weights and bias
        Weights initialized using Xavier initialization
    """
    global fc_layer
    prev_layer_in = input_layer
    weights = []
    biases = []
    for i in layer_sizes:
        weights.append(tf.get_variable("weights_"+str(fc_layer), shape=[prev_layer_in, i], initializer=tf.contrib.layers.xavier_initializer()))
        biases.append(tf.Variable(tf.zeros([i]), name='b_'+str(fc_layer)))
        prev_layer_in = i
        fc_layer += 1
    return weights, biases

def build_cnn_no_pooling(output_layer_size, filter_sizes=[], strides=[], padding=[], fc_sizes=[], weight_decay=0.0001, learning_rate=0.001, transition_sizes=784):
    """ Goes from Conv layers to Conv layers to fc layers, specified by user
        
        Input:
            input_size - input layers dimension
            filter_sizes - array of dimensions of convolution filters
            fc_sizes - size of fc hidden layers at end of the network
            weight_decay - normalization weight decay
            learning_rate - learning rate of optimizer

        Output:
            outputs components of network
    """
    X = tf.placeholder(tf.float32, name='inputs')
    Y = tf.placeholder(tf.float32, name='labels')
    initializer = tf.contrib.layers.xavier_initializer()

    # init the filters
    filters = init_conv_layers(filter_sizes)
    # init the fc layers, with biases
    fc_layers, fc_biases = init_fc_layers(transition_sizes, fc_sizes)

    prev_layer = X
    # convolutional layers
    for i in range(len(filter_sizes)):
        prev_layer = tf.nn.relu(tf.nn.conv2d(prev_layer, filters[i], strides[i], padding[i]))
    
    # fc layers
    # must reshape prev_layer to fit transition
    prev_layer = tf.reshape(prev_layer, [-1, transition_sizes])
    for i in range(len(fc_layers)):
        prev_layer = tf.nn.relu(tf.matmul(prev_layer, fc_layers[i]) + fc_biases[i])
    
    # output layer is classification
    output = tf.nn.softmax(prev_layer)

    # calculate loss
    loss = classification_loss(Y, prev_layer, fc_layers + filters, weight_decay)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    minimizer = optimizer.minimize(loss=loss)
    accuracy = classification_accuracy(Y, output) 

    return X, Y, loss, minimizer, accuracy

def build_linear_model(num_inputs, num_classes, weight_decay, learning_rate):
    """ Uses the number of inputs to build a simple linear model

        Input:
            num_inputs - number of inputs (dimension, flattened)

        Output:
            loss - error in classificiation
            minimizer - needs to be minimized - run this to train
            accuracy - classification accuracy
    """

    X = tf.placeholder(tf.float32, name='inputs')
    Y = tf.placeholder(tf.float32, name='labels')
    initializer = tf.contrib.layers.xavier_initializer()
    input_size = num_inputs
    shape_weights = (input_size, num_classes)
    init = tf.truncated_normal(shape=shape_weights, stddev=3/(num_inputs + num_classes))
    W = tf.get_variable("weights", shape=[input_size, num_classes], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.zeros([num_classes]), name='b')

    y_hat = tf.matmul(X,W) + b
    
    # error loss function, with normalization
    loss = classification_loss(Y, y_hat, [W], weight_decay)

    # update algorithm
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    minimizer = optimizer.minimize(loss=loss)

    # calculate accuracy
    # softmax, get greater than 0.5 - this is it
    accuracy = classification_accuracy(Y, y_hat)

    return X, Y, y_hat, minimizer, accuracy, loss

def train_linear_network(training_data, training_label, test_data, test_label, valid_data, valid_labels):
    # build network
    X, Y, y_hat, mini, acc, loss = build_linear_model(784, 10, 0.001, 0.001)

    batch_size = 500

    # run training, batch size of 7499
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    train_accs = []
    train_losses = []
    test_accs = []
    valid_losses = []
    valid_accs = []
    test_losses = []
    for i in range(100): #n interations
        iters_per_epoch =int(training_data.shape[0]/batch_size) 
        for j in range(iters_per_epoch):
            sess.run(mini, feed_dict={X: training_data[batch_size*j:batch_size*(j+1)],
                                    Y: training_label[batch_size*j:batch_size*(j+1)]})

            # get test loss
            # get training loss
            # get training acc
            # get test loss
        train_acc, train_loss = sess.run([acc, loss], feed_dict={X: training_data,
                                        Y: training_label})
        valid_acc, valid_loss = sess.run([acc, loss], feed_dict={X: valid_data, Y:valid_labels})
        test_acc, test_loss = sess.run([acc, loss], feed_dict={X: test_data, Y: test_label})
        i += iters_per_epoch

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        valid_accs.append(valid_acc)
        valid_losses.append(valid_loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)

    

    # plot losses
    # plot accuracy
    plt.plot(train_accs, label='Training accuracy')
    plt.plot(test_accs, label='Test accuracy')
    plt.plot(valid_accs, label='Valid set accuracy')
    plt.legend()
    plt.show()
    
    plt.plot(train_losses, label='Training losses')
    plt.plot(test_losses, label='Test losses')
    plt.plot(valid_losses, label='Valid set losses')
    plt.legend()
    plt.show()

def one_hot_encode(labels, size):
    print(labels.shape)
    new_b = np.zeros((labels.shape[0], size))
    new_b[np.arange(labels.shape[0]), labels[:]] = 1
    return new_b

def train_deep_network(training, tests, valids):
    training_data, training_labels = training[0], training[1]
    test_data, test_labels = tests[0], tests[1]
    valid_data, valid_labels = valids[0], valids[1]

    N = training_data.shape[1]
    num_classes = training_labels.shape[1]
    filter_sizes = [[7,7,1,3],[3,3,3,1]]
    strides = [[1,1,1,1],[1,1,1,1]]
    padding = ["VALID", "VALID"]
    fc_sizes = [100, num_classes]
    X, Y, loss, minimizer, accuracy = build_cnn_no_pooling(N, fc_sizes=fc_sizes, 
        filter_sizes=filter_sizes, strides=strides, padding=padding,
        weight_decay=0.00001, learning_rate=0.001, transition_sizes=400)

    batch_size = 500
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    train_accs = []
    train_losses = []
    test_accs = []
    valid_losses = []
    valid_accs = []
    test_losses = []
    for i in range(1000): # iterations
        iters_per_epoch =int(training_data.shape[0]/batch_size) 
        for j in range(iters_per_epoch):
            sess.run(minimizer, feed_dict={X: training_data[batch_size*j:batch_size*(j+1)],
                                    Y: training_labels[batch_size*j:batch_size*(j+1)]})

        # get test loss
        # get training loss
        # get training acc
        # get test loss
        train_acc, train_loss = sess.run([accuracy, loss], feed_dict={X: training_data,
                                        Y: training_labels})
        valid_acc, valid_loss = sess.run([accuracy, loss], feed_dict={X: valid_data, Y:valid_labels})
        test_acc, test_loss = sess.run([accuracy, loss], feed_dict={X: test_data, Y: test_labels})
        i += iters_per_epoch

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        valid_accs.append(valid_acc)
        valid_losses.append(valid_loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)

    # plot losses
    # plot accuracy
    plt.plot(train_accs, label='Training accuracy')
    plt.plot(test_accs, label='Test accuracy')
    plt.plot(valid_accs, label='Valid set accuracy')
    plt.legend()
    plt.show()
    
    plt.plot(train_losses, label='Training losses')
    plt.plot(test_losses, label='Test losses')
    plt.plot(valid_losses, label='Valid set losses')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    training_data = extract_data('./data/train-images-idx3-ubyte.gz', 22000)
    training_labels = extract_labels('./data/train-labels-idx1-ubyte.gz', 22000)
    test_data = extract_data('./data/t10k-images-idx3-ubyte.gz', 10000)
    test_labels = extract_labels('./data/t10k-labels-idx1-ubyte.gz', 10000)
    #training_data = training_data.reshape(training_data.shape[0], -1)
    training_labels = training_labels.reshape(-1)
    training_data, valid_data = training_data[:20000], training_data[20000:]
    training_labels = one_hot_encode(training_labels, 10)
    training_labels, valid_labels = training_labels[:20000], training_labels[20000:]
    #test_data = test_data.reshape(test_data.shape[0], -1)
    test_labels = test_labels.reshape(-1)
    test_labels = one_hot_encode(test_labels, 10)

    #show the first 5 images
    for i in range(0):
        plt.imshow(training_data[i].reshape(28,28))
        plt.show()
        print(training_labels[i][:])

    #train_linear_network(training_data, training_labels, test_data, test_labels, valid_data, valid_labels)
    train_deep_network((training_data, training_labels), (test_data, test_labels), (valid_data, valid_labels))

