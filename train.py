import argparse
import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.set_random_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='MNIST')
parser.add_argument("--store_path", type=str, default='')
parser.add_argument("--dev", type=str)
parser.add_argument('--cluster', action='store_true', default=False)
args = parser.parse_args()

import os

if args.dev is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev
import sys
from models import MNIST, C10, C100, STL10
from gan import GAN
from auxnet import AUX

if __name__ == "__main__":
    if args.dataset.upper() == 'MNIST':
        model = MNIST(args.store_path)
    elif args.dataset.upper() == 'C10':
        model = C10(args.store_path)
    elif args.dataset.upper() == 'C100':
        model = C100(args.store_path)
    elif args.dataset.upper() == 'STL10':
        model = STL10(args.store_path)
    else:
        sys.exit("No correct model is given!")

    X, y = model.ds
    with tf.Session() as sess:

        # GAN model
        gan = GAN(model)
        if args.cluster is False:
            gan.train(sess, X, y, model.classes)

        # AuxCluster head
        aux = AUX(model, gan)
        aux.train(sess, X[:y.shape[0]], y, batch=500, epoches=200)
        # aux.classifier(sess, X, y, batch=128, epoches=100)
