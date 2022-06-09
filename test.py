import argparse
import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.set_random_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='MNIST')
parser.add_argument("--store_path", type=str, default='')
parser.add_argument("--dev", type=str)
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
        gan.load_weights(sess)
        # AuxCluster head
        aux = AUX(model, gan)
        aux.load_weights(sess)
        aux.pred(sess, X, y)
