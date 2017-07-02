# DAN: Distributional Adversarial Networks

Tensorflow demo code for paper [Distributional Adversarial Networks](https://arxiv.org/abs/1706.09549) by [Chengtao Li\*](http://chengtaoli.com), [David Alvarez-Melis\*](http://people.csail.mit.edu/davidam/), [Keyulu Xu](http://keyulux.com), [Stefanie Jegelka](http://people.csail.mit.edu/stefje/) and [Suvrit Sra](http://suvrit.de).

---

## Running the Experiments on MNIST

### Training

### Evaluation
To evaluate how well the model recovers the mode frequencies, one need an accurate classifier on MNIST dataset as an approximate label indicator. The code for the classifier is in `mnist_classifier.py` and is adapted from [Tensorflow-Examples](https://github.com/aymericdamien/TensorFlow-Examples/). To train the classifier, run

`python mnist_classifier.py`

After 10 epochs the classifier has an accuracy of \~98\% on test set. The model is stored in the folder `mnist_cnn` for later evaluation.

### Visualization
The following visualization shows how the randomly generated figures evolve through 100 epochs with different models. While for vanilla GAN the figures mostly concentrate on ''easy-to-generate'' modes like `1`, models within DAN framework generate figures that have better coverages over different modes.

|Vanilla GAN                    |  DAN-S                        |  DAN-2S                       |
|:-----------------------------:|:-----------------------------:|:-----------------------------:|
|![](fig/gan.gif "Vanilla GAN") | ![](fig/dan_s.gif "DAN-S")    | ![](fig/dan_2s.gif "DAN-2S")  |


## Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/abs/1706.09549):

```
@article{li2017distributional,
  title={Distributional Adversarial Networks},
  author={Chengtao Li, David Alvarez-Melis, Keyulu Xu, Stefanie Jegelka, Suvrit Sra},
  journal={arXiv preprint arXiv:1706.09549},
  year={2017}
}
```

## Contact
[ctli@mit.edu](mailto:ctli@mit.edu)
