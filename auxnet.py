import tensorflow as tf
import numpy as np
from toolkit import acc, get_weights
from losses import rim, kld
from graphs import aux_net, generate_vat
from sklearn.neighbors import NearestNeighbors


class AUX(object):

    def __init__(self, settings, gan_model):
        self._settings = settings
        self.gan_model = gan_model
        # Placeholders
        self.f_ = tf.placeholder(dtype=tf.float32, shape=[None, gan_model.n_flat], name='flat_')
        self.aux_heads = aux_net(self.f_, settings.n_hidden, settings.classes, settings.n_over_cluster,
                                 settings.n_heads, settings.n_over_heads, 0, settings.std)

        self.aux_var = [var for var in tf.trainable_variables() if "AuxNet" in var.name]
        self._saver = tf.train.Saver(var_list=self.aux_var)

        print('AuxNet n_weights: {}'.format(get_weights(self.aux_var)))

    def train(self, sess, X, y=None, batch=750, epoches=1000):
        r = 5
        opt = tf.train.AdamOptimizer(1e-4)
        f_ = tf.placeholder(dtype=tf.float32, shape=[None, self.gan_model.n_flat])
        val = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        mask_r = tf.tile(val, [r, 1])
        r_norms = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        r_norms_ = tf.tile(r_norms, [r, 1])
        s = self._settings

        # raw, dropped
        f__ = tf.tile(self.f_, [r, 1])
        heads, heads_d = \
            aux_net(f__, s.n_hidden, s.classes, s.n_over_cluster, s.n_heads, s.n_over_heads, 0, s.std), \
            aux_net(f_, s.n_hidden, s.classes, s.n_over_cluster, s.n_heads, s.n_over_heads, 0, s.std)

        # generate r_adv
        '''
        loss = [tf.reduce_sum(kld(heads[i], heads_d[i])) for i in range(len(heads_d))]
        r_adv = tf.gradients(tf.reduce_mean(loss), [f__])[0]
        r_adv = tf.stop_gradient(r_adv)
        r_adv /= (tf.sqrt(tf.reduce_sum(tf.pow(r_adv, 2.0), axis=[1], keep_dims=True)) + 1e-17)
        '''
        r_adv = generate_vat(f__, heads, r_norms_, s, ar=s.ar)
        # '''

        # adv
        x_ = f__ + r_adv * r_norms_ * s.aadv
        head_radv = aux_net(x_, s.n_hidden, s.classes, s.n_over_cluster, s.n_heads, s.n_over_heads, 0, s.std)

        losses = []
        d = [s.d1] * s.n_heads + [s.d2] * s.n_over_heads
        for i in range(len(heads)):
            losses.append(rim(heads[i], heads_d[i], head_radv[i], d[i], mask_r))

        loss = tf.reduce_mean([loss_per_head for loss_per_head in losses])

        # weight_decay = tf.reduce_sum([tf.reduce_sum(tf.abs(w)**2) for w in self.aux_var])
        train_ops = opt.minimize(loss)
        batches_per_epoch = int(np.ceil(X.shape[0] // batch))

        sess.run(tf.global_variables_initializer())
        self.gan_model.load_weights(sess)
        M = self.gan_model(sess, X)
        # '''
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.svm import SVC

        # k = KMeans(n_clusters=10, n_init=20)

        # pred = PCA(n_components=20).fit_transform(M)
        # y_pred = k.fit_predict(pred[:y.shape[0]])
        # print('K-means acc ' + str(acc(y, y_pred[:y.shape[0]])), M.max(0).mean())
        # '''
        # dd = PCA(n_components=20).fit_transform(M)
        # svc = SVC(kernel='linear')
        # svc.fit(dd, y)
        # y_ = svc.predict(dd)
        # print('acc', acc(y, y_))
        trues = np.zeros((M.shape[0], 1))
        trues[:y.shape[0]] = 1
        norms = np.linalg.norm(M, axis=-1, keepdims=True)
        idx = np.arange(M.shape[0])
        ops = [train_ops, loss] + [loss_per_head for loss_per_head in losses]
        scores_ = np.zeros((epoches, s.n_heads))
        for e in range(1, epoches + 1):
            np.random.shuffle(idx)
            stat_loss = []
            M1 = [self.gan_model(sess, X, drop=0.1) for _ in range(r)]
            for i in range(batches_per_epoch):
                idxs = idx[i * batch:(i + 1) * batch]
                tmp = np.concatenate([m1[idxs] for m1 in M1])
                feed = {self.f_: M[idxs], val: trues[idxs], r_norms: norms[idxs], f_: tmp}
                stat_loss.append(sess.run(ops, feed_dict=feed)[1:])

            stat_loss = np.array(stat_loss)
            if e % 1 == 0:
                y_pred = self.pred_(sess, M, self.aux_heads)
                scores = []
                for i in range(s.n_heads):
                    scores.append(acc(y, y_pred[i][:y.shape[0]].argmax(1)))
                    scores_[e - 1, i] = scores[-1]
                    print('AuxHead {} abs acc, loss: {:.2f}%,'
                          ' {:.4f}'.format(i, scores[-1] * 100, np.mean(stat_loss[:, i + 1])))
            print("%-10s  %-15s" % ("Step " + str(e), "M ACC= " + "{:.4f}".format(np.mean(stat_loss[:, 0]))))
            if e % 50 == 0:
                self._saver.save(sess, s.name + '_aux')
        self._saver.save(sess, s.name + '_aux')

    def pred_(self, sess, X, layer):
        y_ = sess.run(layer, feed_dict={self.f_: X})
        return y_

    def classifier(self, sess, X, y=None, batch=128, epoches=1000):

        opt = tf.train.AdamOptimizer(1e-4)
        drop = tf.placeholder(dtype=tf.float32)
        labels = tf.placeholder(dtype=tf.int32, shape=[None])
        from graphs import init_zeros, init_
        with tf.variable_scope('Classifier', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('Dense_1', reuse=tf.AUTO_REUSE):
                w = tf.get_variable('w', shape=[int(self.f_.shape[-1]), 200], initializer=init_)
                b = tf.get_variable('b', shape=[200], initializer=init_zeros)
                l1 = tf.matmul(self.f_, w) + b
                l1 = tf.nn.relu(l1)
                l1 = tf.nn.dropout(l1, rate=drop)

            with tf.variable_scope('Dense_2', reuse=tf.AUTO_REUSE):
                w = tf.get_variable('w', shape=[int(l1.shape[-1]), 10], initializer=init_)
                b = tf.get_variable('b', shape=[10], initializer=init_zeros)
                l2 = tf.matmul(l1, w) + b

        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels, l2))
        train_ops = opt.minimize(loss)

        batches_per_epoch = int(np.ceil(X.shape[0] // batch))

        sess.run(tf.global_variables_initializer())
        self.gan_model.load_weights(sess)
        M = self.gan_model(sess, X)

        idx = np.arange(M.shape[0] - 10000)

        for e in range(1, epoches + 1):
            np.random.shuffle(idx)
            stat_loss = []
            for i in range(batches_per_epoch):
                idxs = idx[i * batch:(i + 1) * batch]
                feed = {self.f_: M[idxs], labels: y[idxs].ravel(), drop: 0.2}
                stat_loss.append(sess.run([train_ops, loss], feed_dict=feed)[1])

            stat_loss = np.array(stat_loss)
            if e % 1 == 0:
                y_pred = sess.run(l2, feed_dict={self.f_: M[-10000:], drop: 0}).argmax(-1)
                print('AuxHead acc {:.2f}'.format((y_pred == y[-10000:].ravel()).sum() / y[-10000:].shape[0]))
            print("%-10s  %-15s" % ("Step " + str(e), "M ACC= " + "{:.4f}".format(np.mean(stat_loss))))

    def load_weights(self, sess):
        self._saver.restore(sess, self._settings.name + '_aux')

    def pred__(self, sess, x, layer, b=10000):
        tmp = [sess.run(layer, feed_dict={self.f_: x[i * b:(i + 1) * b]})
               for i in range(x.shape[0] // b + 1) if x[i * b:(i + 1) * b].shape[0] > 0]
        return np.concatenate(tmp, axis=0)

    def pred(self, sess, X, y):
        s = self._settings

        sess.run(tf.global_variables_initializer())
        self.gan_model.load_weights(sess)
        self.load_weights(sess)
        M = self.gan_model(sess, X)

        y_pred = [self.pred__(sess, M, layer) for layer in self.aux_heads]
        scores = []
        for i in range(s.n_heads):
            scores.append(acc(y, y_pred[i][:y.shape[0]].argmax(1)))
            print('AuxHead {} abs acc, {:.4f}'.format(i, scores[-1] * 100))
