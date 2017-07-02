# DAN: Distributional Adversarial Networks

Tensorflow demo code for paper [Distributional Adversarial Networks](https://arxiv.org/abs/1706.09549) by [Chengtao Li\*](http://chengtaoli.com), [David Alvarez-Melis\*](http://people.csail.mit.edu/davidam/), [Keyulu Xu](http://keyulux.com), [Stefanie Jegelka](http://people.csail.mit.edu/stefje/) and [Suvrit Sra](http://suvrit.de).

---

## Running the Experiments on MNIST
This part of code lies in `mnist` folder and is built based on [DCGAN Implementation](https://github.com/carpedm20/DCGAN-tensorflow).

### Prerequisites
* `Python` 2.7
* `tensorflow` >= 1.0
* `numpy`
* `scipy`
* `matplotlib`

### Training
To train the adversarial network, run
```
python main_mnist.py --model_mode [MODEL_MODE] --is_train True
```
Here `MODEL_MODE` can be one of `gan` (for vanilla GAN model), `dan_s` (for DAN-S) or `dan_2s` (for DAN-2S). 

### Evaluation
To evaluate how well the model recovers the mode frequencies, one need an accurate classifier on MNIST dataset as an approximate label indicator. The code for the classifier is in `mnist_classifier.py` and is adapted from [Tensorflow-Examples](https://github.com/aymericdamien/TensorFlow-Examples/). To train the classifier, run
```
python mnist_classifier.py
```
The classifier has an accuracy of \~97.6\% on test set after 10 epochs and is stored in the folder `mnist_cnn` for later evaluation. To use the classifier to estimate the label frequencies of generated figures, run
```
python main_mnist.py --model_mode [MODEL_MODE] --is_train False
```
The result will be saved to the file specified by `savepath`. A random run gives the following results with different `model_mode`'s.

|              | Vanilla GAN  | DAN-S        | DAN-2S       |
|:------------:|:------------:|:------------:|:------------:|
| Entropy (the higher the better)      | 1.623        | 2.295        | 2.288        | 
| TV Dist (the lower the better)      | 0.461        | 0.047        | 0.061        | 
| L2 Dist (the lower the better)      | 0.183        | 0.001        | 0.003        | 

### Visualization
The following visualization shows how the randomly generated figures evolve through 100 epochs with different models. While for vanilla GAN the figures mostly concentrate on ''easy-to-generate'' modes like `1`, models within DAN framework generate figures that have better coverages over different modes.

|Vanilla GAN                    |  DAN-S                        |  DAN-2S                       |
|:-----------------------------:|:-----------------------------:|:-----------------------------:|
|![](mnist/fig/gan.gif "Vanilla GAN") | ![](mnist/fig/dan_s.gif "DAN-S")    | ![](mnist/fig/dan_2s.gif "DAN-2S")  |


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
Please email to [ctli@mit.edu](mailto:ctli@mit.edu) should you have any questions, comments or suggestions.
