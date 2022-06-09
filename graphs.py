import tensorflow as tf
import numpy as np
from losses import kld, eps

init_ = tf.initializers.random_normal(mean=0., stddev=0.02)
init_zeros = tf.zeros_initializer()


def mnist_discriminator(x, isTraining, drop, drop_last=False, f=32):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        x = sobel(x)

        with tf.variable_scope('CNN_1', reuse=tf.AUTO_REUSE):
            cnn1 = conv2d(x, [4, 4], f, init_, strides=(2, 2))
            cnn1 = tf.layers.batch_normalization(cnn1, 1, scale=True, training=isTraining)
            cnn1 = tf.nn.leaky_relu(cnn1)
            # cnn1 = channel_dropout(cnn1, rate=drop)
            cnn1 = tf.nn.dropout(cnn1, rate=drop)

        with tf.variable_scope('CNN_2', reuse=tf.AUTO_REUSE):
            f *= 2
            cnn2 = conv2d(cnn1, [4, 4], f, init_, strides=(2, 2))
            cnn2 = tf.layers.batch_normalization(cnn2, -1, scale=True, training=isTraining)
            cnn2 = tf.nn.leaky_relu(cnn2)
            # cnn2 = channel_dropout(cnn2, rate=drop)
            cnn2 = tf.nn.dropout(cnn2, rate=drop)

        with tf.variable_scope('CNN_3', reuse=tf.AUTO_REUSE):
            f *= 2
            cnn3 = conv2d(cnn2, [4, 4], f, init_, strides=(2, 2))
            # cnn3 = tf.layers.batch_normalization(cnn3, -1, scale=False, training=isTraining)
            cnn3 = tf.nn.leaky_relu(cnn3)
            if drop_last is True:
                # cnn3 = channel_dropout(cnn3, rate=drop)
                cnn3 = tf.nn.dropout(cnn3, rate=drop)
            flat = tf.reshape(cnn3, (-1, int(np.prod(cnn3.shape[1:]))))

        with tf.variable_scope('Dense', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', shape=[int(flat.shape[-1]), 1], initializer=init_)
            # b = tf.get_variable('b', shape=[1], initializer=init_zeros)
            logit = tf.matmul(flat, w)  # + b

        return flat, logit


def mnist_generator(batch, isTraining, f=128):
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        x = tf.random_uniform((batch, 100), -1, 1)

        with tf.variable_scope('dense', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', shape=[int(x.shape[-1]), f * 3 ** 2], initializer=init_)
            b = tf.get_variable('b', shape=[f * 3 ** 2], initializer=init_zeros)
            dense = tf.matmul(x, w) + b
            cnn1 = tf.reshape(dense, (-1, f, 3, 3))
            cnn1 = tf.nn.leaky_relu(cnn1)

        with tf.variable_scope('CNN_2', reuse=tf.AUTO_REUSE):
            f //= 2
            cnn2 = conv2dT(cnn1, [4, 4], f, init_, strides=(2, 2))
            cnn2 = tf.layers.batch_normalization(cnn2, 1, scale=True, training=isTraining)
            cnn2 = tf.nn.leaky_relu(cnn2)

        with tf.variable_scope('CNN_3', reuse=tf.AUTO_REUSE):
            f //= 2
            cnn3 = conv2dT(cnn2, [4, 4], f, init_, strides=(2, 2))
            cnn3 = tf.layers.batch_normalization(cnn3, 1, scale=True, training=isTraining)
            cnn3 = tf.nn.leaky_relu(cnn3)

        with tf.variable_scope('CNN_4', reuse=tf.AUTO_REUSE):
            f //= 2
            cnn4 = conv2dT(cnn3, [4, 4], f, init_, strides=(2, 2))
            cnn4 = tf.layers.batch_normalization(cnn4, 1, scale=True, training=isTraining)
            cnn4 = tf.nn.leaky_relu(cnn4)

        with tf.variable_scope('CNN_OUT', reuse=tf.AUTO_REUSE):
            f = 1
            cnn5 = conv2d(cnn4, [3, 3], f, init_)
            cnn5 = tf.nn.tanh(cnn5)

        return (tf.transpose(cnn5, [0, 2, 3, 1]) + 1.) / 2.


def cifar_discriminator(x, isTraining, drop, drop_last=False, f=64):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        x = sobel(x)

        # 32 x 32
        with tf.variable_scope('CNN_1', reuse=tf.AUTO_REUSE):
            cnn1 = conv2d(x, [4, 4], f, init_, strides=(2, 2))
            cnn1 = tf.layers.batch_normalization(cnn1, 1, scale=False, training=isTraining)
            cnn1 = tf.nn.leaky_relu(cnn1)
            cnn1 = tf.nn.dropout(cnn1, rate=drop)

        f *= 2
        # 16 x 16
        with tf.variable_scope('CNN_2', reuse=tf.AUTO_REUSE):
            cnn2 = conv2d(cnn1, [4, 4], f, init_, strides=(2, 2))
            cnn2 = tf.layers.batch_normalization(cnn2, 1, scale=False, training=isTraining)
            cnn2 = tf.nn.leaky_relu(cnn2)
            cnn2 = tf.nn.dropout(cnn2, rate=drop)

        f *= 2
        # 8 x 8
        with tf.variable_scope('CNN_3', reuse=tf.AUTO_REUSE):
            cnn3 = conv2d(cnn2, [4, 4], f, init_, strides=(2, 2))
            cnn3 = tf.layers.batch_normalization(cnn3, 1, scale=False, training=isTraining)
            cnn3 = tf.nn.leaky_relu(cnn3)
            cnn3 = tf.nn.dropout(cnn3, rate=drop)

        # 4 x 4
        with tf.variable_scope('Flat', reuse=tf.AUTO_REUSE):
            flat = tf.reshape(cnn3, (-1, int(np.prod(cnn3.shape[1:]))))
            w = tf.get_variable('w', shape=[int(flat.shape[-1]), 1024], initializer=init_)
            b = tf.get_variable('b', shape=[1024], initializer=init_zeros)
            flat = tf.matmul(flat, w) + b
            flat = tf.nn.leaky_relu(flat)
            if drop_last is True:
                flat = tf.nn.dropout(flat, rate=drop)

        # 1024
        with tf.variable_scope('Dense', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', shape=[int(flat.shape[-1]), 1], initializer=init_)
            b = tf.get_variable('b', shape=[1], initializer=init_zeros)
            logit = tf.matmul(flat, w) + b

        return flat, logit


def cifar_generator_t(batch, isTraining, f=512):
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        x = tf.random_uniform((batch, 200), -1, 1)

        c = 3
        with tf.variable_scope('dense', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', shape=[int(x.shape[-1]), f * 4 * 4], initializer=init_)
            b = tf.get_variable('b', shape=[f * 4 * 4], initializer=init_zeros)
            dense = tf.matmul(x, w) + b
            cnn1 = tf.reshape(dense, (-1, f, 4, 4))
            # cnn1 = tf.layers.batch_normalization(cnn1, 1, scale=False, training=isTraining)
            cnn1 = tf.nn.leaky_relu(cnn1)

        with tf.variable_scope('CNN_2', reuse=tf.AUTO_REUSE):
            f //= 2
            cnn2 = conv2dT(cnn1, [4, 4], f, init_, strides=(2, 2))
            cnn2 = tf.layers.batch_normalization(cnn2, 1, scale=False, training=isTraining)
            cnn2 = tf.nn.leaky_relu(cnn2)

        with tf.variable_scope('CNN_3', reuse=tf.AUTO_REUSE):
            f //= 2
            cnn3 = conv2dT(cnn2, [4, 4], f, init_, strides=(2, 2))
            cnn3 = tf.layers.batch_normalization(cnn3, 1, scale=False, training=isTraining)
            cnn3 = tf.nn.leaky_relu(cnn3)

        with tf.variable_scope('CNN_4', reuse=tf.AUTO_REUSE):
            f //= 2
            cnn4 = conv2dT(cnn3, [4, 4], f, init_, strides=(2, 2))
            cnn4 = tf.layers.batch_normalization(cnn4, 1, scale=False, training=isTraining)
            cnn4 = tf.nn.leaky_relu(cnn4)

        with tf.variable_scope('CNN_OUT', reuse=tf.AUTO_REUSE):
            cnn5 = conv2d(cnn4, [3, 3], c, init_)
            cnn5 = tf.nn.tanh(cnn5)
        return (tf.transpose(cnn5, [0, 2, 3, 1]) + 1.) / 2


def stl_discriminator(x, isTraining, drop, drop_last=False, f=64):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        x = sobel(x)

        with tf.variable_scope('CNN_1', reuse=tf.AUTO_REUSE):
            cnn1 = conv2d(x, [4, 4], f, init_, strides=(2, 2))
            cnn1 = tf.layers.batch_normalization(cnn1, 1, scale=False, training=isTraining)
            cnn1 = tf.nn.leaky_relu(cnn1)
            cnn1 = tf.nn.dropout(cnn1, rate=drop)

        with tf.variable_scope('CNN_2', reuse=tf.AUTO_REUSE):
            f *= 2
            cnn2 = conv2d(cnn1, [4, 4], f, init_, strides=(2, 2))
            cnn2 = tf.layers.batch_normalization(cnn2, 1, scale=False, training=isTraining)
            cnn2 = tf.nn.leaky_relu(cnn2)
            cnn2 = tf.nn.dropout(cnn2, rate=drop)

        with tf.variable_scope('CNN_3', reuse=tf.AUTO_REUSE):
            f *= 2
            cnn3 = conv2d(cnn2, [4, 4], f, init_, strides=(2, 2))
            cnn3 = tf.layers.batch_normalization(cnn3, 1, scale=False, training=isTraining)
            cnn3 = tf.nn.leaky_relu(cnn3)
            cnn3 = tf.nn.dropout(cnn3, rate=drop)

        with tf.variable_scope('Flat', reuse=tf.AUTO_REUSE):
            flat = tf.reshape(cnn3, (-1, int(np.prod(cnn3.shape[1:]))))
            w = tf.get_variable('w', shape=[int(flat.shape[-1]), 1024], initializer=init_)
            b = tf.get_variable('b', shape=[1024], initializer=init_zeros)
            flat = tf.matmul(flat, w) + b
            flat = tf.nn.leaky_relu(flat)
            if drop_last is True:
                flat = tf.nn.dropout(flat, rate=drop)

        with tf.variable_scope('Dense', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', shape=[int(flat.shape[-1]), 1], initializer=init_)
            b = tf.get_variable('b', shape=[1], initializer=init_zeros)
            logit = tf.matmul(flat, w) + b

        return flat, logit


def stl_generator_t(batch, isTraining):
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        x = tf.random_uniform((batch, 500), -1, 1)
        f = 512
        hw = 6

        with tf.variable_scope('dense', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', shape=[int(x.shape[-1]), f * hw * hw], initializer=init_)
            b = tf.get_variable('b', shape=[f * hw * hw], initializer=init_zeros)
            dense = tf.matmul(x, w) + b
            cnn1 = tf.reshape(dense, (-1, f, hw, hw))
            cnn1 = tf.nn.leaky_relu(cnn1)

        with tf.variable_scope('CNN_2', reuse=tf.AUTO_REUSE):
            f //= 2
            cnn2 = conv2dT(cnn1, [4, 4], f, init_, strides=(2, 2))
            cnn2 = tf.layers.batch_normalization(cnn2, 1, scale=False, training=isTraining)
            cnn2 = tf.nn.leaky_relu(cnn2)

        with tf.variable_scope('CNN_3', reuse=tf.AUTO_REUSE):
            f //= 2
            cnn3 = conv2dT(cnn2, [4, 4], f, init_, strides=(2, 2))
            cnn3 = tf.layers.batch_normalization(cnn3, 1, scale=False, training=isTraining)
            cnn3 = tf.nn.leaky_relu(cnn3)

        with tf.variable_scope('CNN_4', reuse=tf.AUTO_REUSE):
            f //= 2
            cnn4 = conv2dT(cnn3, [4, 4], f, init_, strides=(2, 2))
            cnn4 = tf.layers.batch_normalization(cnn4, 1, scale=False, training=isTraining)
            cnn4 = tf.nn.leaky_relu(cnn4)

        with tf.variable_scope('CNN_OUT', reuse=tf.AUTO_REUSE):
            f = 3
            cnn6 = conv2d(cnn4, [4, 4], f, init_)
            cnn6 = tf.nn.tanh(cnn6)
        return (tf.transpose(cnn6, [0, 2, 3, 1]) + 1.) / 2


def dna_discriminator(x, isTraining, drop, drop_last=False, f=32):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        # x = sobel(x, sobel=True)

        with tf.variable_scope('CNN_1', reuse=tf.AUTO_REUSE):
            cnn1 = conv2d(x, [4, 8], f, init_, strides=(2, 4))
            cnn1 = tf.layers.batch_normalization(cnn1, 1, scale=False, training=isTraining)
            cnn1 = tf.nn.leaky_relu(cnn1)
            cnn1 = tf.nn.dropout(cnn1, rate=drop)

        with tf.variable_scope('CNN_2', reuse=tf.AUTO_REUSE):
            f *= 2
            cnn2 = conv2d(cnn1, [4, 4], f, init_, strides=(2, 2))
            cnn2 = tf.layers.batch_normalization(cnn2, 1, scale=False, training=isTraining)
            cnn2 = tf.nn.leaky_relu(cnn2)
            cnn2 = tf.nn.dropout(cnn2, rate=drop)

        with tf.variable_scope('CNN_3', reuse=tf.AUTO_REUSE):
            f *= 2
            cnn3 = conv2d(cnn2, [3, 4], f, init_, strides=(1, 2))
            cnn3 = tf.layers.batch_normalization(cnn3, 1, scale=False, training=isTraining)
            cnn3 = tf.nn.leaky_relu(cnn3)
            cnn3 = tf.nn.dropout(cnn3, rate=drop)

        with tf.variable_scope('Flat', reuse=tf.AUTO_REUSE):
            flat = tf.reshape(cnn3, (-1, int(np.prod(cnn3.shape[1:]))))
            w = tf.get_variable('w', shape=[int(flat.shape[-1]), 1024], initializer=init_)
            b = tf.get_variable('b', shape=[1024], initializer=init_zeros)
            flat = tf.matmul(flat, w) + b
            flat = tf.nn.leaky_relu(flat)
            if drop_last is True:
                flat = tf.nn.dropout(flat, rate=drop)

        with tf.variable_scope('Dense', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', shape=[int(flat.shape[-1]), 1], initializer=init_)
            # b = tf.get_variable('b', shape=[1], initializer=init_zeros)
            logit = tf.matmul(flat, w)  # + b

        return flat, logit


def dna_generator_t(batch, isTraining):
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        x = tf.random_uniform((batch, 100), -1, 1)
        f = 512
        hw = 3 * 8

        with tf.variable_scope('dense', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', shape=[int(x.shape[-1]), f * hw], initializer=init_)
            b = tf.get_variable('b', shape=[f * hw], initializer=init_zeros)
            dense = tf.matmul(x, w) + b
            cnn1 = tf.reshape(dense, (-1, f, 3, 8))
            cnn1 = tf.nn.leaky_relu(cnn1)

        with tf.variable_scope('CNN_2', reuse=tf.AUTO_REUSE):
            f //= 2
            cnn2 = conv2dT(cnn1, [3, 4], f, init_, strides=(1, 2))
            cnn2 = tf.layers.batch_normalization(cnn2, 1, scale=False, training=isTraining)
            cnn2 = tf.nn.leaky_relu(cnn2)

        with tf.variable_scope('CNN_3', reuse=tf.AUTO_REUSE):
            f //= 2
            cnn3 = conv2dT(cnn2, [4, 4], f, init_, strides=(2, 2))
            cnn3 = tf.layers.batch_normalization(cnn3, 1, scale=False, training=isTraining)
            cnn3 = tf.nn.leaky_relu(cnn3)

        with tf.variable_scope('CNN_4', reuse=tf.AUTO_REUSE):
            f //= 2
            cnn4 = conv2dT(cnn3, [4, 8], f, init_, strides=(2, 4))
            cnn4 = tf.layers.batch_normalization(cnn4, 1, scale=False, training=isTraining)
            cnn4 = tf.nn.leaky_relu(cnn4)

        with tf.variable_scope('CNN_OUT', reuse=tf.AUTO_REUSE):
            f = 8
            cnn6 = conv2d(cnn4, [3, 3], f, init_)
            cnn6 = tf.nn.tanh(cnn6)
        return cnn6


def aux_net(x_, n_hidden=1024, n_classes=10, n_over_clusters=50, n_clusterheads=1,
            n_over_heads=1, drop=0., std=0.001):
    heads = []
    N = [n_classes] * n_clusterheads + [n_over_clusters] * n_over_heads
    nn = n_hidden
    fn_init_ = tf.initializers.random_normal(mean=0, stddev=std)

    with tf.variable_scope('AuxNet', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('Dense_1', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', shape=[int(x_.shape[-1]), nn], initializer=fn_init_)
            b = tf.get_variable('b', shape=[nn], initializer=init_zeros)
            l1 = tf.matmul(x_, w) + b
            l1 = tf.nn.relu(l1)
            l1 = tf.nn.dropout(l1, rate=drop)

        with tf.variable_scope('Dense_2', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', shape=[int(l1.shape[-1]), nn], initializer=fn_init_)
            b = tf.get_variable('b', shape=[nn], initializer=init_zeros)
            l2 = tf.matmul(l1, w) + b
            l2 = tf.nn.relu(l2)
            l2 = tf.nn.dropout(l2, rate=drop)

        for i in range(len(N)):
            with tf.variable_scope('Cluster_Head_' + str(i + 1)):
                w = tf.get_variable('w', shape=[int(l2.shape[-1]), N[i]], initializer=fn_init_)
                b = tf.get_variable('b', shape=[N[i]], initializer=init_zeros)
                hs = tf.matmul(l2, w) + b
                heads.append(tf.nn.softmax(hs))

    return heads


def generate_vat(x_, hs, norms_, settings, ar=1):
    x_ = tf.stop_gradient(x_)
    d = tf.random_normal(shape=tf.shape(x_))
    d /= (tf.sqrt(tf.reduce_sum(tf.pow(d, 2.0), axis=[1], keep_dims=True)))
    aa = x_ + d * norms_ * ar
    hs_ = aux_net(aa, settings.n_hidden, settings.classes, settings.n_over_cluster,
                  settings.n_heads, settings.n_over_heads, 0, settings.std)
    loss = [tf.reduce_mean(kld(hs[i], hs_[i])) for i in range(len(hs_))]
    radv = tf.gradients(tf.reduce_mean(loss), [x_])[0]
    radv = tf.stop_gradient(radv)
    radv /= (tf.sqrt(tf.reduce_sum(tf.pow(radv, 2.0), axis=[1], keep_dims=True)) + eps)
    return radv


def conv2d(x, kernel, channels, init_, strides=(1, 1), pad='SAME'):
    w = tf.get_variable('w', shape=kernel + [x.shape.as_list()[1], channels], initializer=init_)
    return tf.nn.conv2d(x, w, strides=(1, 1,) + strides, padding=pad, data_format='NCHW')


def conv2dT(x, kernel, channels, init_, strides=(2, 2), pad='SAME'):
    w = tf.get_variable('w', shape=kernel + [channels, x.shape.as_list()[1]], initializer=init_)
    out = [x.shape.as_list()[0], channels] + [x.shape.as_list()[-2] * strides[0], x.shape.as_list()[-1] * strides[1]]
    return tf.nn.conv2d_transpose(x, w, out, strides=(1, 1,) + strides, padding=pad, data_format='NCHW')


def sobel(x):
    with tf.variable_scope('Sobel'):
        channels = x.shape.as_list()[1]
        gray = x
        if channels == 3:
            w = tf.reshape(tf.constant([0.2989, 0.5870, 0.1140], tf.float32), [1, 1, 3, 1])
            gray = tf.nn.conv2d(gray, w, [1, 1, 1, 1], padding='SAME', data_format='NCHW')
        sobel_w = tf.constant([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], tf.float32, name='sobels_w')
        dx = tf.reshape(sobel_w, [3, 3, 1, 1])
        dy = tf.transpose(dx, [1, 0, 2, 3])
        sobels = tf.concat((dx, dy), -1)
        sobels_xy = tf.nn.depthwise_conv2d(gray, sobels, [1] * 4, padding='SAME', name='Sobel', data_format='NCHW')
        x = (x - 0.5) / 0.5
        con = tf.concat((sobels_xy, x), 1) # if sobel else x
    return con


'''
def sobel(x):
    with tf.variable_scope('Sobel'):
        gray = (x + 1) / 2
        channels = x.shape.as_list()[-1]
        if channels == 3:
            w = tf.reshape(tf.constant([0.2989, 0.5870, 0.1140], tf.float32), [1, 1, 3, 1])
            gray = tf.nn.conv2d(gray, w, [1, 1, 1, 1], padding='SAME')
        sobel_x = tf.constant([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], tf.float32, name='sobels_x')
        dx = tf.reshape(sobel_x, [3, 3, 1, 1])
        dy = tf.transpose(dx, [1, 0, 2, 3])
        sobels = tf.concat((dx, dy), -1)
        sobels_xy = tf.nn.depthwise_conv2d(gray, sobels, [1, 1, 1, 1], padding='SAME', name='Sobel')
        con = tf.concat((sobels_xy, x), -1)
    return con

def custom_cnn(x, kernel, channels, init_, strides=(1, 1), pad='SAME'):
    w = tf.get_variable('w', shape=kernel + [x.shape.as_list()[-1]] + [channels], initializer=init_)
    # b = tf.get_variable('b', shape=[1] + [channels] + [1, 1], initializer=tf.initializers.zeros())
    return tf.nn.conv2d(x, w, strides=(1,) + strides + (1,), padding=pad, data_format='NHWC')  # + b


def custom_cnn_transpose(x, kernel, channels, init_, strides=(2, 2), pad='SAME'):
    w = tf.get_variable('w', shape=kernel + [channels] + [x.shape.as_list()[-1]], initializer=init_)
    # b = tf.get_variable('b', shape=[1] + [channels] + [1, 1], initializer=tf.initializers.zeros())
    out = [x.shape.as_list()[0]] + [x.shape.as_list()[-3] * strides[0], x.shape.as_list()[-2] * strides[1]] + [channels]
    return tf.nn.conv2d_transpose(x, w, out, strides=(1,) + strides + (1,), padding=pad, data_format='NHWC')  # + b


def cifar_generator_t(batch, isTraining):
    with tf.variable_scope('Adversarial', reuse=tf.AUTO_REUSE):
        x = tf.random.uniform((batch, 200), -1, 1)
        f = 512
        c = 3
        with tf.variable_scope('dense', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', shape=[int(x.shape[-1]), f * 4 * 4], initializer=init_)
            b = tf.get_variable('b', shape=[f * 4 * 4], initializer=init_zeros)
            dense = tf.matmul(x, w) + b
            cnn1 = tf.reshape(dense, (-1, 4, 4, f))
            cnn1 = tf.layers.batch_normalization(cnn1, -1, scale=False, training=isTraining)
            cnn1 = tf.nn.leaky_relu(cnn1)

        with tf.variable_scope('CNN_2', reuse=tf.AUTO_REUSE):
            f //= 2
            cnn2 = custom_cnn_transpose(cnn1, [4, 4], f, init_, strides=(2, 2))
            cnn2 = tf.layers.batch_normalization(cnn2, -1, scale=False, training=isTraining)
            cnn2 = tf.nn.leaky_relu(cnn2)

        with tf.variable_scope('CNN_3', reuse=tf.AUTO_REUSE):
            f //= 2
            cnn3 = custom_cnn_transpose(cnn2, [4, 4], f, init_, strides=(2, 2))
            cnn3 = tf.layers.batch_normalization(cnn3, -1, scale=False, training=isTraining)
            cnn3 = tf.nn.leaky_relu(cnn3)

        with tf.variable_scope('CNN_4', reuse=tf.AUTO_REUSE):
            f //= 2
            cnn4 = custom_cnn_transpose(cnn3, [4, 4], f, init_, strides=(2, 2))
            cnn4 = tf.layers.batch_normalization(cnn4, -1, scale=False, training=isTraining)
            cnn4 = tf.nn.leaky_relu(cnn4)

        with tf.variable_scope('CNN_OUT', reuse=tf.AUTO_REUSE):
            cnn5 = custom_cnn(cnn4, [3, 3], c, init_)
            cnn5 = tf.nn.tanh(cnn5)
        return cnn5


def cifar_discriminator(x, isTraining, drop, drop_last=False):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        x = sobel(tf.transpose(x,[0,1,2,3]))
        f = 64
        # 32 x 32
        with tf.variable_scope('CNN_1', reuse=tf.AUTO_REUSE):
            cnn1 = custom_cnn(x, [4, 4], f, init_, strides=(2, 2))
            cnn1 = tf.layers.batch_normalization(cnn1, -1, scale=False, training=isTraining)
            cnn1 = tf.nn.leaky_relu(cnn1)
            cnn1 = tf.nn.dropout(cnn1, rate=drop)

        f *= 2
        # 16 x 16
        with tf.variable_scope('CNN_2', reuse=tf.AUTO_REUSE):
            cnn2 = custom_cnn(cnn1, [4, 4], f, init_, strides=(2, 2))
            cnn2 = tf.layers.batch_normalization(cnn2, -1, scale=False, training=isTraining)
            cnn2 = tf.nn.leaky_relu(cnn2)
            cnn2 = tf.nn.dropout(cnn2, rate=drop)

        f *= 2
        # 8 x 8
        with tf.variable_scope('CNN_3', reuse=tf.AUTO_REUSE):
            cnn3 = custom_cnn(cnn2, [4, 4], f, init_, strides=(2, 2))
            cnn3 = tf.layers.batch_normalization(cnn3, -1, scale=False, training=isTraining)
            cnn3 = tf.nn.leaky_relu(cnn3)
            cnn3 = tf.nn.dropout(cnn3, rate=drop)

        # 4 x 4
        with tf.variable_scope('Flat', reuse=tf.AUTO_REUSE):
            flat = tf.reshape(cnn3, (-1, int(np.prod(cnn3.shape[1:]))))
            w = tf.get_variable('w', shape=[int(flat.shape[-1]), 1024], initializer=init_)
            b = tf.get_variable('b', shape=[1024], initializer=init_zeros)
            flat = tf.matmul(flat, w) + b
            flat = tf.nn.leaky_relu(flat)
            if drop_last is True:
                flat = tf.nn.dropout(flat, rate=drop)

        # 1024
        with tf.variable_scope('Dense', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', shape=[int(flat.shape[-1]), 1], initializer=init_)
            b = tf.get_variable('b', shape=[1], initializer=init_zeros)
            logit = tf.matmul(flat, w) + b

        return flat, logit


'''
