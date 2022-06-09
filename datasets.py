import tensorflow as tf
import numpy as np
import cv2
from sys import platform

stl10_path = './stl10'

img_to_array, array_to_img = tf.keras.preprocessing.image.img_to_array, tf.keras.preprocessing.image.array_to_img


def image_resize(x, dim=(64, 64)):
    tmp = np.zeros(list(x.shape[:1]) + list(dim) + list(x.shape[-1:]), dtype=x.dtype)
    for i in range(x.shape[0]):
        aa = cv2.resize(x[i], dsize=dim, interpolation=cv2.INTER_LANCZOS4)  # INTER_CUBIC
        tmp[i] = aa if len(aa.shape) == 3 else np.expand_dims(aa, -1)
    return tmp


def STL10_DS(resize=48):
    x_train = np.fromfile(stl10_path + 'train_X.bin', dtype=np.uint8)
    x_train = x_train.reshape((int(x_train.size / 3 / 96 / 96), 3, 96, 96)).transpose((0, 3, 2, 1))

    x_unlabeled = np.fromfile(stl10_path + 'unlabeled_X.bin', dtype=np.uint8)
    x_unlabeled = x_unlabeled.reshape((int(x_unlabeled.size / 3 / 96 / 96), 3, 96, 96)).transpose((0, 3, 2, 1))

    x_test = np.fromfile(stl10_path + 'test_X.bin', dtype=np.uint8)
    x_test = x_test.reshape((int(x_test.size / 3 / 96 / 96), 3, 96, 96)).transpose((0, 3, 2, 1))

    x_train = image_resize(x_train, (resize, resize))
    x_unlabeled = image_resize(x_unlabeled, (resize, resize))
    x_test = image_resize(x_test, (resize, resize))

    y_train = np.fromfile(stl10_path + 'train_y.bin', dtype=np.uint8) - 1
    y_test = np.fromfile(stl10_path + 'test_y.bin', dtype=np.uint8) - 1

    x_train = (np.concatenate((x_train, x_test, x_unlabeled), 0))
    y_train = np.concatenate((y_train, y_test), 0)

    return x_train, y_train


def C10_DS():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = np.concatenate((x_train, x_test), 0)
    y_train = np.concatenate((y_train, y_test), 0)
    return x_train, y_train


def C100_DS():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')
    x_train = np.concatenate((x_train, x_test), 0)
    y_train = np.concatenate((y_train, y_test), 0)
    return x_train, y_train


def MNIST_DS():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.concatenate((x_train, x_test), 0)
    y_train = np.concatenate((y_train, y_test), 0)
    x_train = np.expand_dims(x_train[:, 2:26, 2:26], -1)
    return x_train, y_train


def build_dataset(x, batch_size, epoch):
    with tf.device("/cpu:0"):
        ds = tf.data.Dataset.from_tensor_slices({'x': x})
        ds = ds.repeat(epoch)
        ds = ds.shuffle(10 * batch_size, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(30)
        iterator = ds.make_initializable_iterator()
        images = iterator.get_next()
        return images, iterator
