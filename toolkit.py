import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_scatter(pred, labels=None, title=None):
    plt.figure(figsize=(12, 10))

    colors = plt.cm.nipy_spectral(np.arange(labels.max() + 1) / (labels.max() + 1))
    cm = LinearSegmentedColormap.from_list(
        'mylist', colors, N=(labels.max() + 1))
    ##aa =plt.cm.get_cmap('cubehelix', labels.max() + 1)
    if labels is not None:
        plt.scatter(pred[:, 0], pred[:, 1], c=labels, cmap=cm)
        plt.colorbar(ticks=np.arange(labels.max() + 1))
    else:
        plt.scatter(pred[:, 0], pred[:, 1])
    if title is not None:
        plt.title(title)
        plt.savefig(title + '.png')

    # plt.xlim(-20, 20)
    # plt.ylim(-20, 20)
    plt.show()


def acc_detailed(y_true, y_pred, class_num=10):
    def warn(*args, **kwargs):
        pass

    import warnings
    warnings.warn = warn
    from sklearn.utils.linear_assignment_ import linear_assignment
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # print('print w:')
    # print(w)
    ind = linear_assignment(w.max() - w)
    # print('print ind:')
    # print(ind)
    for j in range(class_num):
        index = np.where(ind[:, 1] == j)
        i = ind[index, 0]
        pre_true_num = w[i, j]
        real_true_num = np.sum(y_true == j)
        # print('class: %d, acc rate: %7.4f' % (j, 1.0 * pre_true_num / real_true_num))
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, [ind, w]


def acc(y_true, y_pred):
    def warn(*args, **kwargs):
        pass

    import warnings
    warnings.warn = warn
    from scipy.optimize import linear_sum_assignment as linear_assignment

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size


def get_weights(weights):
    n = 0
    for w in weights:
        n += np.product(w.shape)
    return n
