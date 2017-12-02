from tflearn.datasets import * 
import numpy as np 

def load_mnist():
    x_train, y_train, x_test, y_test = mnist.load_data(data_dir="./data/mnist/")
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    return x_train, y_train 

def load_flowers(resize_pics=(64, 64)):
    X, Y = oxflower17.load_data(dirname="./data/17flowers/", resize_pics=resize_pics, shuffle=True, one_hot=False)
    # X = X[Y < 10]
    return X, Y

def load_cifar10():
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data(dirname="./data/cifar-10-batches-py/", one_hot=False)
    return X_train, Y_train


def batch_iter(data, batch_size, total_batches, shuffle=True):
    num_epochs = int(total_batches / (data.shape[0] / float(batch_size)))
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    load_mnist()
    load_flowers()
    load_cifar10()
