# Flower Classification with TPU #

In this [r, we use Convolutional Neural Network to classify flowers into different classes. For more information on the data-set, see TensorFlow flowers [data set](https://www.tensorflow.org/datasets/catalog/tf_flowers). 
In order to speed up classificaiton, we make use of Tensor Processing Units (TPU), which are available from [Kaggle](https://www.kaggle.com/docs/tpu). 

We consider different augmentations to the data-set, and different regularisation and learning rates for the pre-trained models.


### TPU

For training these models with TPU, we use a context manger which tells TensorFlow how to divide the work of training amoung the 8 TPU cores.


## Model Design 

We make use of *transfer learning*, where we use a pretrained model and append additional fully connected layers on the end so we can exploit the pretrained "learnt" representations, and then just have to learn how to map these learnt representations to classifications. We may consider more models, but at this stage, we consider:

- VGGNet [(Simonyan and Zisserman, 2014)](https://arxiv.org/abs/1409.1556)
- ResNet [(He et al, 2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
- Xception [(François Chollet, 2017)](https://arxiv.org/abs/1610.02357)
- DenseNet [(Huang et al, 2016)](https://arxiv.org/abs/1608.06993)

These models were pretrained on the [ImageNet](http://image-net.org/) dataset.

### Reguralisation Techniques
While training the networks we use early stopping and apply a learning rate scheduler.

(TODO):
- Add Data Augmentations


### Metrics and Loss

We use cross-entropy as the loss function. The reason we use `sparse_categorical_crossentropy` as opposed to `categorical_crossentropy` is since the responses that are not hot-one encodings, e.g. [0,0,1], but rather as numbers, e.g. [3]. 

What are precision and recall in this context?


## Final Predictions

The final model used is an ensemble of ResNet and Xception.
We achieved an F1 score of 0.833
