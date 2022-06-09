# Image Clustering Using an Augmented Generative Adversarial Network and Information Maximization

## Introduction
**This is an implementation code written in Python (version 3.6.9) of a manuscript [paper](https://ieeexplore.ieee.org/document/9451540)
**

## Performance

The reported performance of our proposed model is based on custom architecture.<br>
We train our unsupervised proposed method for 5 independent runs on training and testing set and we report the results accordingly. 

### Best and Average Performance

Below table reports the best and average recorded performance from our model.

Dataset | Best Acc | Aver Acc
--- | --- | ---
MNIST|99.02% | 98.85% (±0.14%)
CIFAR-10|70.04% | 69.22% (±0.83%)
CIFAR-100/20 | 32.44% | 30.88% (±0.14%)
STL10| 58.65 % | 74.7% (±1.81%)

## Usage

All hyper-parameters of the reported accuracy are stored in 'models.py'. To run the training code.

#### MNIST

```
python train.py --dataset mnist --store_path ./models/mnist/mnist.ckpt
```

#### CIFAR-10

To run the training code.

```
python train.py --dataset c10 --store_path ./models/c10/c10.ckpt
```

#### CIFAR-100/20

To run the training code.

```
python train.py --dataset c100 --store_path ./models/c100/c100.ckpt
```

#### STL10

To run the training code.

```
python train.py --dataset stl10 --store_path ./models/stl10/stl10.ckpt
```

## The evaluation of the model.

##### Example evaluation on MNIST, CIFAR10/100-20 & STL10:

Through the argument '--store_path', the full path of the stored model is parsed.

```
python test.py --dataset mnist --store_path ./models/mnist/mnist.ckpt
```

```
python test.py --dataset c10 --store_path ./models/c10/c10.ckpt
```

```
python test.py --dataset c100 --store_path ./models/c100/c100.ckpt
```

```
python test.py --dataset stl10 --store_path ./models/stl10/stl10.ckpt
```

## Notes

- The classifier head is trained and evaluated only for labelled set on STL10 dataset. The unlabelled part of STL10 is
  used only to train the GAN model.

- All tests have been performed in Cuda version 10.1.

## Citation

```shell
@ARTICLE{9451540,
  author={Ntelemis, Foivos and Jin, Yaochu and Thomas, Spencer A.},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Image Clustering Using an Augmented Generative Adversarial Network and Information Maximization}, 
  year={2021},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TNNLS.2021.3085125}}
```


