# Flower Classification with TPU #

We use Convolutional Neural Network to classify flowers into different classes. For more information 
on the data-set, see TensorFlow flowers [data set](https://www.tensorflow.org/datasets/catalog/tf_flowers). 
In order to speed up classificaiton, we make use of Tensor Processing Units (TPU), which are 
available from [Kaggle](https://www.kaggle.com/docs/tpu). 

We consider different augmentations to the data-set, and different regularisation and learning 
rates for the pre-trained models.


### TPU

For training these models with TPU, we use a context manger which tells TensorFlow how to divide 
the work of training among the 8 TPU cores.


## Model Design 

We make use of *transfer learning*, where we use a pretrained model and append additional fully 
connected layers on the end so we can exploit the pretrained "learnt" representations, and then 
just have to learn how to map these learnt representations to classifications. We may consider 
more models, but at this stage, we consider:

- VGGNet [(Simonyan and Zisserman, 2014)](https://arxiv.org/abs/1409.1556)
- ResNet [(He et al, 2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
- Xception [(François Chollet, 2017)](https://arxiv.org/abs/1610.02357)
- DenseNet [(Huang et al, 2016)](https://arxiv.org/abs/1608.06993)
- EfficientNet [(Mingxing and Quoc, 2019)](https://arxiv.org/abs/1905.11946)

Most of these models were pretrained on the [ImageNet](http://image-net.org/) dataset.

### Reguralisation Techniques
While training the networks we use early stopping and apply a learning rate scheduler.

### Data Augmentation Techniques
- Random blackout augmentation (Adding random blacked out regions within the image to remove some information)
- Image augmentations
  - flip: randomly flip along vertical axis
  - rotate: rotate by k*90 where k is chosen uniformly from {0,1,2,3}
  - contrast: randomly adjust the difference between darks and lights 
  - saturate: randomly adjusts the ["colourfulness"](https://en.wikipedia.org/wiki/Colorfulness) of the image 
  - hue: randomly adjusts the [hue](https://en.wikipedia.org/wiki/Hue ) of the image 


### Metrics and Loss
We use cross-entropy as the loss function since this is a classification problem.
The reason we use `sparse_categorical_crossentropy` as opposed to `categorical_crossentropy` is 
since the responses that are not hot-one encodings, e.g. [0,0,1], but rather as numbers, 
e.g. [3]. 


## Final Predictions

The final model used is an ensemble of DenseNet and EfficientNet with a final score of 0.95756 
which places me in the top 15 of the leaderboards.

My leaderboard scores are:


| Score                 | Date (version) | Notes                                                          |
|-----------------------|----------|----------------------------------------------------------------|
| 0.76018               | 23-10-22 | ResNet50V2 wih no regularisation                               | 
| 0.83895               | 26-10-22 | ResNet50V2 with dropout of 0.1038                              | 
| **0.95756**           | 31-10-22 | Ensemble of EfficientNet and DenseNet (No Regularization)      | 
