import tensorflow as tf

eps = 1e-16


def entropy(p):
    p = tf.clip_by_value(p, eps, 1. - eps)
    return -tf.reduce_sum(p * tf.log(p), axis=-1)


def rim(hs, hs_d, hs_radv, d, mask=None):
    k = tf.cast(hs.shape[1], tf.float32)
    mask = tf.tile(mask, [1, k])
    hs, hs_d, hs_radv = tf.boolean_mask(hs, mask), tf.boolean_mask(hs_d, mask), tf.boolean_mask(hs_radv, mask)
    hs, hs_d, hs_radv = tf.reshape(hs, (-1, k)), tf.reshape(hs_d, (-1, k)), tf.reshape(hs_radv, (-1, k))
    px = tf.reduce_mean(hs, 0)
    u = tf.ones_like(px, dtype=tf.float32) / k
    marg = tf.nn.relu(kld(px, u) - tf.log(k) * d)
    ent = tf.reduce_mean(entropy(hs))

    kld_aug = tf.reduce_mean(kld(hs, hs_d)) * 0.5
    hs = tf.stop_gradient(hs)
    kld_adv = tf.reduce_mean(kld(hs, hs_radv)) * 0.5

    im = 4 * marg + ent
    kld_ = kld_aug + kld_adv
    loss = im * 0.2 + kld_
    return loss


def kld(p_logit, q_logit):
    p_logit = tf.clip_by_value(p_logit, eps, 1.)
    q_logit = tf.clip_by_value(q_logit, eps, 1.)
    return tf.reduce_sum(p_logit * (tf.log(p_logit) - tf.log(q_logit)), axis=-1)
