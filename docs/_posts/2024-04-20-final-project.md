---
title: "Final Project"
date: 2024-04-20
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

In our project, we aim to analyze the performance of classification models
trained on data synthesized/augmented using Stable Diffusion and ControlNet,
specifically in low-data scenarios. In this analysis, we investigate the
optimal dataset size for the application of such synthetic image generation,
and which of several different image-generation techniques produces a training
dataset that most improves the performance of an image classifier. We hope that
our experiments can contribute to a broader standing of generative models and
further elucidate their applications in data augmentation.

<img src="{{ site.base_url }}{% link /assets/imgs/sd_examples.png %}"/>

<center>
  <em>
    Examples of Stable Diffusion, a text-to-image model.
  </em>
</center>

<img src="{{ site.base_url }}{% link /assets/imgs/controlnet_examples.jpg %}"/>

<center>
  <em>
    Examples of ControlNet on different types of spatial conditioning inputs.
  </em>
</center>

## Related Works

ControlNet [1] is a neural network architecture that improves upon
text-to-image diffusion models by allowing for better control over these
models. Diffusion models are conditioned on control signals (e.g. edges, depth,
segmentation) that represent desired attributes, allowing users to fine tune
the generation process. We use ControlNet extensively throughout our project
for data augmentation purposes.

[2] describes work done by a student at another university on data augmentation
using Stable Diffusion and ControlNet. They were able to achieve an average
F1-score increase of 0.85 percentage points on the MS-COCO dataset after
extending two classes with synthetic training data. While this was very similar
to the problem we are trying to solve, a much more complex image-generation
pipeline was implemented using CLIP and BLIP to generate text for a ControlNet
prompt, which we believe can be simplified. Additionally, this work relies on a
large dataset of 80 classes, and we believe data augmentation will be most
practical and effective on a simpler task with less pre-existing data.

ResNet [3] is the model we are training for image classification due to its
ease of use and training, and its prevalence throughout the field of image
classification. ResNet solves the vanishing gradient problem in deep neural
networks using skip connections that bypass layers during backpropagation.

[4] brings up methods of simulating dataset imbalance on CIFAR dataset. They
removed 99% of cat images from CIFAR-10 and used synthetic methods (GAN/stable
diffusion) to address the dataset imbalance. They found that synthetic methods
performed much better than traditional augmentation methods (e.g. color jitter,
perspective shift, rotation). However, they noticed that synthetic methods had
trouble consistently creating data that represented the original distribution.
The dataset imbalance described in this work was only for a single class, but
we intend to discuss the extent to which image generation can improve accuracy
across the entire dataset. Additionally, the benefits of stable diffusion
specifically are hindered by the small image format, since stable diffusion
operates best at 512x512, but the CIFAR-10 only consists of 32x32 images, so
the stable diffusion images had to be downsized significantly.

[5] provides a novel means of image generation for use specifically as
synthetic data for an image classifier, and demonstrates the efficacy of the
method on the CIFAR and ImageNet datasets. Showing SOTA results in the area we
are experimenting in, the intricacies of their method would certainly be
interesting to explore, but they largely dismiss traditional means of image
generation applied to data augmentation due to the significant compute needed.
We intend to test the simplest image-generation pipeline possible in order to
maximize the practicality of traditional methods, and see if we can obtain
results anywhere near a training-aware method.

## Methods / Approach

#### <u>Initial Experiment</u>

For our initial experiments, we decided to augment datasets with a supplemental
synthetic dataset created solely by Stable Diffusion 1. This augmented dataset
consisted of 1000 classes based on those of the ImageNet-1K dataset, a commonly
used benchmark in image classification. For each class, we generated 20 512x512
images, approximately doubling the size of each class, with Stable Diffusion
using only the class name as the prompt. This resulted in each class being
generated in a variety of positions and background environments, which was
ideal for the augmentation we were trying to achieve with this synthetic
dataset. We then observed how a ResNet-50 classifier trained from scratch on a
real image dataset performed against a ResNet-50 classifier trained from
scratch on the combined dataset of the existing real dataset and synthetic
dataset. We found that the ResNet-50 model trained on the combined dataset
performed better than the model trained just on the real image dataset across
all of our accuracy metrics, including an f1 accuracy of 0.066 compared to
0.025.

#### <u>Changes to Initial Approach</u>

From our initial investigation, we had a few key takeaways:

1. The ImageNet-1K dataset presents too difficult of a task for a model without
   pretraining. Additionally, with 1000 classes, generating enough augmented
   data to meaningfully impact the dataset across all classes is impractical.
2. The quality of images from Stable Diffusion is highly variable based on the
   image size and the class being generated. Stable Diffusion is most effective
   at 512x512, and cannot easily understand what certain classes refer to.

<img src="{{ site.base_url }}{% link /assets/imgs/sd-robin.png %}"/>

<center>
  <em>
    Stable Diffusion image confusing Robin bird with Robin from Magic the Gathering
  </em>
</center>

Based on this, we decided to make several changes in order to provide a more
realistic context for and improve the quality of our synthetic data:

1. Going forward, we used the Intel Image Classification dataset which has just
   over 4000 images for each of 6 different classes representing various
   natural scenes (e.g. “forest”, “building”, “sea”, etc.). The smaller number of
   classes presents an easier task to train classes on, and allows us to isolate
   specific classes in specific experiments.
2. We used ControlNet rather than just Stable Diffusion to generate images.
   Doing so provides another vector of control over image generation, the
   control image input. We tried two different types of control images, both based
   on images from the original dataset: segmented and canny-edge.
3. Rather than training from scratch, we used a pretrained ResNet-50 for our
   new experiments. This simulates how the synthetic dataset would perform for
   real image-classification tasks.

Despite synthesized images not being completely representative of real images,
our initial experiments show that adding these images can improve the accuracy
of image classifiers on real-image datasets in very low-data cases. Differing
from previous works, we intend to test the efficacy of supplementing an
existing dataset with synthetic data at various dataset sizes, in order to find
the point adding synthetic data is most effective and most ineffective. Our
methods are largely similar to previous works, yet our initial experiments show
that minimal image synthesis already performs well in low data-applications.
Unlike most other works, we intend to keep the image generation pipeline as
simple as possible, and focus more on what single type of control image and
prompt will improve performance most.

#### <u>ControlNet Experiments Overview</u>

Image generation using ControlNet, every image in the dataset was first
processed with both canny edge-detection or object segmentation before being
used as the control image. For canny edge-detection, the opencv canny
edge-detector was applied first. For segmentation, UPerNet object segmentation
was used to determine all the distinct object regions in an image. We
considered using something more modern such as SegmentAnything, but the UPerNet
object single-image output was very simple to work with and closely aligned
with the ControlNet segmentation input. After each image was processed,
ControlNet used the processed image along with a prompt of just the image class
to generate a new image. Two new images were generated for each image in the
original dataset, one corresponding to the segmented image and one
corresponding to the canny edge image. We separated these new images into two
datasets, one consisting of the original dataset combined with all the
segmentation-generated images (`seg` dataset) and the other consisting of the
original dataset combined with all the canny-generated images (`canny`
dataset). This dataset resulted in the most promising results, so we created
one additional dataset: we first generated one image off of every image in the
original dataset using ControlNet canny edge-detection just as before, except
we used a prompt of the class name along with “4k photo HD”. We then combined
the original dataset with these new canny-generated images to form our third
new dataset (`canny2` dataset).

<img src="{{ site.base_url }}{% link /assets/imgs/three-cnet-images.png %}"/>

<center>
  <em>
    Example images from the three datasets that were created using ControlNet.
  </em>
</center>

## Experiment Setup

As aforementioned, we are able to generate three new datasets using ControlNet
(`seg`, `canny`, and `canny2`), all of which derive from the original training
dataset of the Intel Image Classification dataset. Additionally, it would be
interesting to explore different scenarios regarding the sparsity of data.
Considering that the Intel Image dataset contains approximately 14,000 images
within the train dataset in total, we would like to see specifically how the
model fares if we limit the training data to only consist of 1/20, 1/50, and
1/100 of the original dataset (≈700, 280, and 140 total images, respectively;
≈117, 47, and 23 images per class, respectively).

We have two independent variables (IV) we are exploring: (1) the type of
augmentation we will employ onto our dataset, and (2) the fraction of data
preserved for model training to simulate data sparsity. Below are the levels of
each independent variable:

- (IV 1) Augmentation Type
  - None (`raw`)
  - Standard, randomized image transformations (`tfms`)
  - `seg`
  - `canny`
  - `canny2`
- (IV 2) Fraction of Preserved Training Data
  - Full
  - 1/20
  - 1/50
  - 1/100

This yields a 5x4 factorial experimental design, a total of 20 different models
for our experiment. Data sparsity is simulated by first randomly shuffling
before splitting the training dataset and keeping the portion that is the size
of our desired fraction; in this sense, we mimic randomly sampling the
appropriate fraction of the training data. We added another level to IV 1,
`tfms`, because it is a common practice in the computer vision domain to apply
random image transformations to the training dataset, thus it would be
particularly interesting to observe its performance versus augmentation via
ControlNet-generated data. For the generative augmentation types (`seg`,
`canny`, `canny2`), the size of the training dataset would be doubled from the
current IV 2 level (e.g. the `canny` & 1/20 condition pair will result in the
model’s training data size being 1/10 of the original, since 1/20 would be
“randomly sampled” from the original dataset, and 1/20 is “randomly sampled”
from the generated dataset with the same methodology as discussed earlier).

Regardless of IV levels, all models take in as input a color image representing
a natural scene, and outputs a probability vector for each of 6 possible
classes specified in the Intel dataset (“buildings”, “forest”, “glacier”,
“mountain”, “sea”, “street”). Our validation split, to be used for model
evaluation during the training loop, was unconventionally derived from randomly
splitting half of the testing subset of the Intel dataset instead of the
training subset. This was done to simplify the process of ensuring generated
images stay in the training split and to make it much easier to reproduce
results in our experiment. The remaining half of the testing subset (our test
split) was reserved for evaluation after training is complete against three
different metrics: recall, precision, and f1-score. We used the Scikit-Learn
library to assist with computing these accuracy metrics.

<img src="{{ site.base_url }}{% link /assets/imgs/intel-mountain-ex.png %}"/>

<center>
  <em>
    Example image in the Intel dataset for the model to classify
  </em>
</center>

We opted to use the PyTorch and PyTorch Lightning frameworks for our
experiments. All trained classification models for each condition pair utilized
the ResNet50 backbone which is frozen during training along with two fully
connected layers to perform the actual classification from the features
extracted by the backbone. The Adam optimizer is utilized with a learning rate
of `1e-5`, and all models were trained for a maximum of 50 epochs with an early
stopping mechanism that monitors the validation cross entropy loss (patience =
5). We derive these hyperparameters mainly from trial and error from
preliminary experimentation of the model and the dataset.

#### <u>Side Experiment: Stable Diffusion 1.5</u>

Out of curiosity, we performed an experiment with Stable Diffusion. We wanted
to test the performance of a model trained purely on Stable Diffusion images
for one label. We removed all 2191 images corresponding to the buildings label
from the Intel Image Classification dataset, which has the fewest images of all
6 labels. We then used Stable Diffusion 1.5
([`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
from HuggingFace) to generate 2500 images. Initially, these images were
generated with only the prompt “Building”. After training on these images, we
found that they lacked sufficient diversity to properly represent some of the
buildings in the original Intel Image Classification dataset. Thus, we
generated these images again with the following prompts: “City building”,
“Religious building”, “Rural building”, “Modern building”, “Historic building”,
“Industrial building”, “Residential building”, “Commercial building”,
“Skyscraper”, and “Townhouse”. These prompts were picked at random, with a 50%
chance of appending an “s” to each prompt after selection. The additional “s”
was appended to ensure that Stable Diffusion did not always generate images of
a singular building but also had cityscapes. This set of data was trained with
the same configuration described above.

## Results

Previously, [2] performed a similar experiment, although on a different dataset
(6 classes from MS-COCO, 50% of training data because of time constraints).
They saw little to no improvements using Stable Diffusion (average of 0.0031
improvement over baseline for F1 score over 6 classes). Similarly, they also
used ControlNet as an augmentation, with focus on canny edge and segmentation
with minimal to none improvements over baseline (range of -0.007 to +0.0038).
Finally, they saw significantly worse performance for a classifier trained
solely on synthetic data (average of 11.37 decrease over baseline). Although
many of these results are not directly comparable to ours because of
significantly different experiment conditions, we can see a similar trend of
minimal improvement using these data augmentation methods especially for the
full dataset ([2] did not experiment with using fractional amounts of the
original dataset).

<img src="{{ site.base_url }}{% link /assets/imgs/5x4results.png %}"/>

Our two baselines were `raw` and `tfms`, for the original dataset and the
augmented dataset with standard transformations. Interestingly, `raw`
performed better than `tfms`. This is likely due to overfitting, especially
with the smaller dataset sizes making the model more prone to remember the data
with their augmentations.

We found that for the reduced-size experiments, ControlNet augmentation
achieved better performance than the baselines. For the full training data,
the augmentations resulted in slightly worse, but comparable performance to the
original data. An interesting result is that `canny` on 1/100th of the data
(85.3%) produced better results than `raw` on 1/50th of the data (84.1%). This
means that for small datasets, ControlNet has the potential to improve the
generalization of classification models better than additional data collection.
As we deal with larger datasets, the benefit of ControlNet decreases, although
it is still better than standard augmentations.

Specifically, `canny`, or Canny ControlNet conditioning with the label as the
prompt, was the best augmentation scheme. `canny2` with the more refined
prompt followed closely behind. We believe the prompt caused ControlNet to
generate a specific, more colorful style which resulted in poor generalization
in classification. `seg` performs noticeably worse, and we believe the vaguer
conditioning input compared to Canny increased the variance of the dataset too
much. ControlNet tended to generate wilder results, sometimes physically
implausible. Below is one example of a glacier in the dataset. There is
oversaturation from the refined prompt in Canny 2, and an unusual sky in
segmentation from the lack of edge detail used in Canny conditioning. The
basic Canny result looks the most real, while maintaining the overall shape of
the original.

<img src="{{ site.base_url }}{% link /assets/imgs/cnet-augs.png %}"/>

Note that across all ControlNet augmentations, there was a bias in the most
“typical” example of the label. Above, while the original glacier has rock and
dust on it, the ControlNet augmentations all show pristine glaciers, much like
what a human would first picture upon the word “glacier”. This bias is caused
by the training data of ControlNet and Stable Diffusion, and is a major
weakness of using generative augmentation. They are also prone to generating
art instead of photos. For this dataset, even when specifying “photo” in the
prompt of `canny2`, there were a lot of sketches in the augmentations, like
this example of a street.

<img src="{{ site.base_url }}{% link /assets/imgs/cnet-street.jpg %}"/>

While there are various weaknesses, our ControlNet augmentation approach worked
well for small datasets, particularly with basic Canny conditioning. This
shows the potential for diffusion-based augmentation methods to be used for
situations where data is hard to obtain.

#### <u>Side Experiment: Stable Diffusion 1.5</u>

|        | Recall    | Precision | F1        |
| ------ | --------- | --------- | --------- |
| Run #1 | 0.830     | 0.862     | 0.816     |
| Run #2 | **0.869** | **0.886** | **0.865** |

The second run with more diverse prompts to Stable Diffusion 1.5 yielded
significantly better results, with an F1 score of 0.865 compared to 0.816.
Comparing this to the full raw dataset F1 score of 0.931, the F1 score from
completely removing a label and augmenting with Stable Diffusion is lower. This
is expected, as many of the Stable Diffusion images for buildings were not
representative of real buildings. For instance, the building below has many
warped balconies.

<img src="{{ site.base_url }}{% link /assets/imgs/sd-building.jpg %}"/>

<center>
  <em>
    Example of building image generated with Stable Diffusion.
  </em>
</center>

Overall, this suggests that data quality is very important. Completely
supplementing a label with images generated with Stable Diffusion is not
viable. Instead, Stable Diffusion should be used as an augmentation technique,
as suggested by improvements that we saw by using Stable Diffusion on
ImageNet-1K.

## Discussion

In this project, we experimented with two relatively new methods for data
augmentation backed by generative AI: Stable Diffusion and ControlNet. We were
able to test these data augmentation methods across many different experiment
settings using different augmentation methods and varying amounts of the
original dataset. We were able to expand our understanding of the capabilities
and limitations of data augmentation. In particular, it’s important to note
that while data augmentation can be a good choice in improving a model’s
performance and generalization abilities, it may not always be ideal and may
even make model performance worse in some cases. Although the amount of data is
certainly important, more data often results in diminishing returns and calls
for higher quality data instead.

In our project, we were only able to test two datasets, ImageNet-1K and Intel
Image Classification. It’s likely that our results are heavily influenced by
these datasets, and as such, an avenue of future work would be to perform these
experiments on different datasets. Additionally, it may be interesting to
investigate how the tuning of hyperparameters for the Stable Diffusion /
ControlNet models can affect the results.

## Challenges Encountered

Throughout this project, our team encountered many different challenges. One
such challenge was dataset selection. Initially, we chose ImageNet-1K–this
dataset has 1000 classes, so it was difficult to distinguish whether the
improvements that we saw by augmenting with Stable Diffusion were actually
effective. Additionally, it was difficult to find datasets that were compatible
with the synthetic images because of various constraints such as image size.
Thus, we moved to the Intel Image Classification dataset, which only had 6
classes. A slight issue that we had with this dataset was that the labels were
pretty easily distinguishable (baseline ResNet50 already scored very well on
this dataset), so it was difficult to see improvements by applying data
augmentations.

If we were to start over today, our additional experience from completing this
project would lead us to be more careful in our dataset selection. Being hasty
in our initial dataset selection caused us to have to switch our dataset midway
through the project, which had a noticeable impact on the time we had to
conduct our experiments.

## References

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

[5] Huang, J. Liu, S. You, and C. Xu, “Active Generation for Image
Classification.” Available: https://arxiv.org/pdf/2403.06517.pdf
