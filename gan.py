import tensorflow as tf
import numpy as np
from toolkit import acc, get_weights
from datasets import build_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time


class GAN(object):

    def __init__(self, settings):
        self._settings = settings

        # Placeholder
        self.x_ = tf.placeholder(dtype=tf.uint8, shape=settings.input, name='x_')
        x__ = tf.transpose(tf.cast(self.x_, tf.float32) / 255., [0, 3, 1, 2])
        # x__ = (tf.cast(self.x_, tf.float32) - 127.5) / 127.5
        self.train_mode = tf.placeholder(dtype=tf.bool)
        self.drop_rate = tf.placeholder(dtype=tf.float32)

        # Generator
        self.fake_ = settings.generator_graph(settings.batch, self.train_mode)

        # Discriminator
        self.flat, self.logit = settings.discriminator_graph(x__, self.train_mode, self.drop_rate, drop_last=True)
        self.n_flat = self.flat.shape[-1]

        # Create list of the weights
        self.disc_var = [var for var in tf.all_variables() if "Discriminator" in var.name]
        self.gen_var = [var for var in tf.all_variables() if "Generator" in var.name]
        self._saver = tf.train.Saver(var_list=self.disc_var + self.gen_var)

        print('Discriminator n_weights: {}'.format(get_weights(self.disc_var)))
        print('Generator n_weights: {}'.format(get_weights(self.gen_var)))

    def train(self, sess, X, y=None, n_clusters=10):

        with tf.device('/cpu:0'):
            images, iterator = build_dataset(self.x_, self._settings.batch, int(1e+10))

        images['x'] = tf.transpose(tf.cast(images['x'], tf.float32) / 255., [0, 3, 1, 2])
        _, real_logit = self._settings.discriminator_graph(images['x'], True, 0.2, drop_last=True)
        dw = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if real_logit.name.split('/')[0] in var.name]

        fake_ = self._settings.generator_graph(self._settings.batch, True)

        fake__ = tf.transpose(fake_, [0, 3, 1, 2])
        __, fake_logit = self._settings.discriminator_graph(fake__, True, 0.2, drop_last=True)
        dw += [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if fake_logit.name.split('/')[0] in var.name]

        opt = tf.train.AdamOptimizer(2e-4, beta1=0.5)
        aw = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if fake_.name.split('/')[0] in var.name]
        train_disc_var = [var for var in tf.trainable_variables() if "Discriminator" in var.name]
        train_adv_var = [var for var in tf.trainable_variables() if "Generator" in var.name]

        losses_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit, labels=tf.ones_like(real_logit))
        losses_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.zeros_like(fake_logit))
        d_loss = tf.reduce_mean(losses_real) + tf.reduce_mean(losses_fake)

        lim = 20
        con = tf.concat((_, __), 0)
        pen_pos = tf.reduce_sum(tf.nn.relu(con - lim))
        pen_neg = tf.reduce_sum(tf.nn.relu(-con / 0.2 - lim))
        reg = pen_pos + pen_neg

        g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.ones_like(fake_logit))
        g_loss = tf.reduce_mean(g_loss)

        d_train = opt.minimize(d_loss + reg * 1e-3, var_list=train_disc_var)
        g_train = opt.minimize(g_loss, var_list=train_adv_var)

        d_train = tf.group([d_train, dw])
        g_train = tf.group([g_train, aw])
        name = self._settings.name
        sess.run(iterator.initializer, feed_dict={self.x_: X})
        sess.run(tf.global_variables_initializer())

        k = KMeans(n_clusters=n_clusters, n_init=20)
        pca = PCA(n_components=20)
        stats = []
        for e in range(1, self._settings.epoches + 1):
            loss_d, loss_g = [], []
            aa = time.time()
            for i in range(500):
                loss_d.append(sess.run([d_train, d_loss])[-1])
                loss_g.append(sess.run([g_train, g_loss])[-1])
            if e % 50 == 0 and e > 0 and y is not None:
                sss = self.pred(sess, X, self.flat)
                pred = pca.fit_transform(sss)
                y_pred = k.fit_predict(pred[:y.shape[0]])
                stats.append([e, acc(y, y_pred[:y.shape[0]]), np.mean(loss_d), np.mean(loss_g)])
                # np.save('./records/' + self._settings.name, stats)
                print('K-means acc ' + str(stats[-1][1]), sss.max(0).mean())

            print("%-10s %-15s %-15s %-15s" % ("Step " + str(e), "D ACC= " + "{:.4f}".format(np.mean(loss_d)),
                                               "G ACC= " + "{:.4f}".format(np.mean(loss_g)),
                                               "{:.4f}".format(time.time() - aa)))
        self._saver.save(sess, name)

    def pred(self, sess, x, layer=None, d=0., b=10000):
        layer = self.flat if layer is None else layer
        tmp = [sess.run(layer, feed_dict={self.x_: x[i * b:(i + 1) * b], self.drop_rate: d, self.train_mode: False})
               for i in range(x.shape[0] // b + 1) if x[i * b:(i + 1) * b].shape[0] > 0]
        return np.concatenate(tmp, axis=0)

    def load_weights(self, sess):
        self._saver.restore(sess, self._settings.name)

    def __call__(self, sess, x, drop=0, **kwargs):
        return self.pred(sess, x, d=drop)
