from graphs import mnist_discriminator, mnist_generator, dna_generator_t, dna_discriminator, \
    cifar_discriminator, cifar_generator_t, stl_discriminator, stl_generator_t
from datasets import MNIST_DS, C10_DS, C100_DS, STL10_DS


class MNIST:
    def __init__(self, name='./models/mnist'):
        self.input = [None, 24, 24, 1]
        self.classes = 10
        self.batch = 128
        self.epoches = 500
        self.n_hidden = 1024
        self.n_over_cluster = 50
        self.n_over_heads = 1
        self.n_heads = 5
        self.discriminator_graph = mnist_discriminator
        self.generator_graph = mnist_generator
        self.ds = MNIST_DS()
        self.name = name.lower()
        self.std = 0.01
        self.d1 = 0.0001
        self.d2 = 0.01
        self.ar = 1.
        self.aadv = 0.2


class CIFAR:
    def __init__(self):
        self.input = [None, 32, 32, 3]
        self.batch = 128
        self.epoches = 1000
        self.discriminator_graph = cifar_discriminator
        self.generator_graph = cifar_generator_t
        self.std = 0.001
        self.d1 = 0.0001
        self.d2 = 0.01
        self.ar = 0.3
        self.aadv = 0.15


class C10(CIFAR):
    def __init__(self, name='./models/c10'):
        super(C10, self).__init__()
        self.classes = 10
        self.ds = C10_DS()
        self.name = name
        self.n_over_cluster = 50
        self.n_over_heads = 1
        self.n_heads = 5
        self.n_hidden = 1024


class C100(CIFAR):
    def __init__(self, name='./models/c100'):
        super(C100, self).__init__()
        self.classes = 20
        self.ds = C100_DS()
        self.name = name
        self.n_over_cluster = 100
        self.n_over_heads = 1
        self.n_heads = 5
        self.n_hidden = 1024


class STL10:
    def __init__(self, name='./models/stl10'):
        self.input = [None, 48, 48, 3]
        self.classes = 10
        self.batch = 128
        self.epoches = 2000
        self.discriminator_graph = stl_discriminator
        self.generator_graph = stl_generator_t
        self.ds = STL10_DS()
        self.name = name
        self.n_over_cluster = 50
        self.n_over_heads = 1
        self.n_heads = 5
        self.n_hidden = 1024
        self.std = 0.01
        self.d1 = 0.0001
        self.d2 = 0.01
        self.ar = 0.3
        self.aadv = 0.15
