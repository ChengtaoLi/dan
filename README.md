# DAN: Distributional Adversarial Networks

Tensorflow demo code for paper [Distributional Adversarial Networks](https://arxiv.org/abs/1706.09549) by [Chengtao Li](http://chengtaoli.com), [David Alvarez-Melis](http://people.csail.mit.edu/davidam/), [Keyulu Xu](http://keyulux.com), [Stefanie Jegelka](http://people.csail.mit.edu/stefje/) and [Suvrit Sra](http://suvrit.de).

---

## Running the Experiments on MNIST

### Training

### Evaluation


### Visualization
The following visualization shows how the randomly generated figures evolve through 100 epochs with different models. While for vanilla GAN the figures mostly concentrate on ''easy-to-generate'' modes like 1 and 7, models within DAN framework generate figures that have better coverages over different modes.

![Vanilla GAN](fig/gan.gif)

![DAN-S](fig/dan_s.gif)

![DAN-2S](fig/dan_2s.gif)

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
