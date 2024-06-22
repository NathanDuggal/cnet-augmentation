---
title: "Project Update"
date: 2024-03-26
layout: post
---

## Introduction

Data augmentation is an important technique in artificial intelligence for
increasing the diversity and size of training datasets, allowing for improved
model performance and generalization. Traditional data augmentation techniques
for computer vision involve transformations such as flipping, rotating, or
scaling. However, because these techniques are simple, they may fail to capture
complex details that would improve model generalization abilities.

Recent advancements in generative models include Stable Diffusion and
ControlNet. Stable Diffusion is a generative AI model that creates images from
a text prompt. Based on Stable Diffusion, ControlNet takes in a control image
in addition to a text prompt and generates a new image based on the prompt and
control, which can be used to modify existing images or create something
completely new. These models provide new avenues for data augmentation and help
alleviate issues including class imbalance or limited data availability.

In our project, we aim to analyze the performance of classification and
segmentation models trained on data synthesized/augmented using Stable
Diffusion and ControlNet. We hope that our experiments can contribute to a
broader standing of generative models and further elucidate their applications
in data augmentation.

<img src="{{ site.base_url }}{% link /assets/imgs/sd_examples.png %}"/>
*Examples of Stable Diffusion, a text-to-image model.*

<img src="{{ site.base_url }}{% link /assets/imgs/controlnet_examples.jpg %}"/>
*Examples of ControlNet on different types of spatial conditioning inputs.*

## Related Works

ControlNet [1] is a neural network architecture that improves upon
text-to-image diffusion models by allowing for better control over these
models. Diffusion models are conditioned on control signals (e.g. edges, depth,
segmentation) that represent desired attributes, allowing users to fine tune
the generation process. We plan to use ControlNet closer to the final part of
our project for data augmentation purposes.

[2] describes work done by a student at another university on data augmentation
using Stable Diffusion and ControlNet. They were able to achieve an average
F1-score increase of 0.85 percentage points on the MS-COCO dataset after
extending two classes with synthetic training data.

[3] describes ResNet, which is the model we are training for image
classification. ResNet solves the vanishing gradient problem in deep neural
networks using skip connections that bypass layers during backpropagation.

[4] brings up methods of simulating dataset imbalance on CIFAR dataset. They
removed 99% of cat images from CIFAR-10 and used synthetic methods (GAN/stable
diffusion) to address the dataset imbalance. They found that synthetic methods
performed much better than traditional augmentation methods (e.g. color jitter,
perspective shift, rotation). However, they noticed that synthetic methods had
trouble consistently creating data that represented the original distribution.

#### References

[1] L. Zhang and M. Agrawala, “Adding Conditional Control to Text-to-Image
Diffusion Models,” arXiv.org, Feb. 10, 2023. https://arxiv.org/abs/2302.05543

[2] O. Melin, R. Rynell, L. Ha, and K. Wojtulewicz, “Generating Synthetic
Training Data with Stable Diffusion”, 2024. Available:
https://liu.diva-portal.org/smash/get/diva2:1779399/FULLTEXT01.pdf

[3] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image
Recognition,” Dec. 2015. Available: https://arxiv.org/pdf/1512.03385.pdf

[4] T. Eliassen and Y. Ma, “Data Synthesis with Stable Diffusion for Dataset
Imbalance -Computer Vision.” Available:
https://cs230.stanford.edu/projects_fall_2022/reports/17.pdf

## Methods / Approach

For our initial experiments, we decided to augment datasets with a supplemental synthetic dataset created solely by Stable Diffusion 1. This augmented dataset consisted of 1000 classes based on those of the ImageNet-1K dataset, a commonly used benchmark in image classification. For each class, we generated 20 512x512 images with Stable Diffusion using only the class name as the prompt. This resulted in each class being generated in a variety of positions and background environments, which was ideal for the augmentation we were trying to achieve with this synthetic dataset. We then observed how an image classifier trained on a real image dataset performed against a classifier trained on the combined dataset of the existing real dataset and synthetic dataset. We also attempted training a model on just the synthetic data, but we found that this was too little data to result in reasonable performance. Creating a larger volume of synthetic data would allow us to effectively perform such experiments, and is a goal of ours before the final report. We experimented with creating a larger image volume in two ways already: other image-generation models  and batching. 

Although Stable Diffusion 1 worked relatively well, we also investigated using Stable Diffusion 2. Using Stable Diffusion 2 to generate images was found to take up to 20x longer per image compared to Stable Diffusion 1, and because the images from Stable Diffusion 1 were not of significantly lower quality, we decided to proceed with Stable Diffusion 1 for the duration of our experiments. Performance improvements towards Stable Diffusion may become necessary with a larger image volume or more intricate model like ControlNet, so using xformers or other transformer accelerators will be integrated into our process before the final submission. 

Image generation was conducted on the PACE ICE. Parallelization of image generation was attempted using batch scripting, but this was not used in the final image generation process. Using batching would likely greatly increase the volume of generated images significantly, which could allow us to test larger and other more specific data imbalances, and will likely be necessary if we use a more computationally expensive image-generation model for our final experiments. Batching will be investigated further for the final report. 


## Experiments / Results

The dataset we chose for our procedure was a “minified” version of the original
ImageNet-1K dataset, named ImageNet Mini. Like ImageNet-1K, ImageNet Mini by
default comes in predetermined `train/` and `val/` folders. However, for the
purposes of evaluating our model, we designated the validation folder as our
test split, while we performed a 90-10 random split on the training folder as
our training and validation splits, respectively. Furthermore, unlike
ImageNet-1K, ImageNet Mini only has approximately 30 training examples per
class (in `train/`). This mimics the sparse-data scenario that we are
hypothesizing that an image generator like Stable Diffusion would be most
advantageous in.

We settled with training ResNet50 on ImageNet Mini as a baseline model. We did
not include any pre-trained weights for this model, since the available weights
were derived from the ImageNet-1K dataset, which would defeat the purposes of
our investigation. We utilized the `torchvision` implementation of ResNet50,
while we used a combination of PyTorch and PyTorch Lightning for our model
training and evaluation implementations. Then, we retrained another ResNet50
model from scratch on an augmented ImageNet Mini dataset with images generated
by Stable Diffusion, which adds approximately 20 new training images to each
class. Note that these images are added to `train/` before the train-validation
split, while `val/` is left unmodified.

| Condition (Val Split) | Top 3 Acc | Top 5 Acc | Precision | Recall | F1     |
| --------------------- | --------- | --------- | --------- | ------ | ------ |
| ImageNet Mini (Raw)   | .18077    | .23518    | .08559    | .08348 | .06557 |
| ImageNet Mini + SD    | .58693    | .66264    | .48493    | .41508 | .40372 |

| Condition (Test Split) | Top 3 Acc  | Top 5 Acc  | Precision  | Recall     | F1         |
| ---------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| ImageNet Mini (Raw)    | .08233     | .11420     | .02906     | .03696     | .02546     |
| ImageNet Mini + SD     | **.15167** | **.20469** | **.07879** | **.07724** | **.06602** |

Firstly, the baseline model trained on the original ImageNet Mini yielded very
poor initial performance. However, considering that there are only a few
examples relative to the amount of classes in the dataset (~30 examples for
each of 1000 classes), these results were somewhat expected. We may also have
used an early stopping algorithm that is too sensitive in our training process,
which could also cause this subpar performance. Finally, comparing the test
metrics from the validation split metrics suggest that the test split (`val/`) is
noticeably more difficult to classify compared to the other splits.

Nonetheless, the addition of Stable Diffusion-generated images into the dataset
seem to have noticeably increased the accuracy of the baseline ResNet50 model
across the board. All measured metrics (top 3/5 accuracy, precision, recall,
and f1-score) have increased after the introduction of SD-generated images,
with more dramatic increases in the validation split. We realized this was an
inaccurate metric, because the train/validation split was randomized without
the same seed between training and inference. Thus, the incorrect validation
statistics more closely align with the training statistics.

However, top 3/5 test accuracy with augmentation is nearly 2 times that of
without augmentation. This shows that the augmented data from SD allowed the
model to learn and capture features of each class more thoroughly. Because the
test split did not include any generated images, we can see that the features
from SD are rather generalizable, and it is potentially a viable option in
problems with little data and class imbalance.

## What’s Next & Team Member Contributions

We first hope to fix the issue with the train/validation split. Since we will
be working with different augmentation experiments, the process of retrieving
the data splits based on augmentation method should be more streamlined too.

We will also devote significant time to testing augmentations with ControlNet.
A good initial experiment is running ControlNet on basic spatial conditioning
input with the prompt as the class name. Basic conditioning includes edge
detectors and normal maps, because they require no training.

Then, we will take a pretrained segmentation model to feed into ControlNet.
Segment Anything and YOLO are examples of SOTA segmentation models we may use.
On top of vanilla segmentation, we will try incorporating a binary mask
separating foreground from background. In this case, the foreground is the
segmented areas for the ground truth class. Our idea is to allow ControlNet to
create images with a similar subject but completely different background. There
are many extensions of this technique we will try if time permits.

<img src="{{ site.base_url }}{% link /assets/imgs/segment_anything.png %}"/>
*Examples of the Segment Anything Model.*

To augment our dataset with ControlNet, we have multiple experiments we are
interested in. We will first augment all of the images in the original ImageNet
mini dataset using ControlNet with a binary mask filter to form a completely
new augmented dataset. We then will compare the performance of a model trained
on the original ImageNet mini to a model trained on just the augmented dataset.
Additionally, we will also test the performance of a model trained on the
combined ImageNet mini and augmented datasets, and if time permits we can test
different mixtures of original and augmented images to determine the optimal
extent of the augmentation.

We will also further refine our testing procedures. The ResNet50 model we used
in our initial experiments did not align very well with our relatively small
initial dataset size, so we may use an alternative model and increase the size
of our dataset.

| Task                                                                               | Completion Date | Team Member     |
| ---------------------------------------------------------------------------------- | --------------- | --------------- |
| Fix and streamline train/val/test split process with augmentation                  | 3/31            | Adrian, Daniel  |
| Pretrained segmentation model with basic input (SAM/YOLO)                          | 3/31            | Richard, Daniel |
| ControlNet augmentation with basic input (edge detector)                           | 4/7             | Adrian, Richard |
| Integrate segmentation into ControlNet input                                       | 4/14            | Adrian, Richard |
| Test ControlNet augmentations on ResNet                                            | 4/14            | Nathan, Richard |
| Test other variations of segmentation-based input for ControlNet (if time permits) | 4/18            | Adrian, Daniel  |
| Further refine testing procedures (if time permits)                                | 4/18            | Nathan, Richard |
| Final Report                                                                       | 4/18            | Everyone        |
