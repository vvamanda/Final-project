# Clear-and-Rainy-Transformation-Based-on-CycleGAN-Model
The goal of this project is to use a CycleGAN model to perform the conversion between rainy images and clear images. 
# Introduction
With the continuous development of autonomous driving technology, driving scenarios in bad weather conditions have been one of the important challenges faced by autonomous driving systems. Severe weather conditions such as rain, snow or haze can seriously affect sensor performance and reduce image quality, resulting in reduced reliability for critical tasks such as target detection, road recognition and vehicle tracking. Therefore, in order to improve the robustness of the autopilot system in bad weather conditions, we try to implement a rainy-clear conversion. 

We will explore in depth the design principle of CycleGAN model and its performance in rainy-clear conversion tasks. Through the research of this project, we hope to provide new insights and solutions for the further application of rainy-clear conversion technology in intelligent transportation systems. Ultimately, we expect this work to lead to substantial improvements in the reliability and safety of autonomous driving systems in a variety of weather conditions.
# File Description
We have a total of five code folders and two of our main files: test.py and train.py.

## train.py
General-purpose training script for image-to-image translation.

It first creates model, dataset, and visualizer given the option.

It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
```
python train.py --dataroot ./dataset/rainy_to_clear --name rainy_to_clear --model cycle_gan --use_wandb
```
## test.py
General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.
```
python test.py --dataroot ./dataset/rainy_to_clear/testA --name rainy_to_clear --model test --no_dropout
```
```
python test.py --dataroot ./dataset/rainy_to_clear/testB --name rainy_to_clear --model test --no_dropout
```
## Others
`data`    This package includes all the modules related to data loading and preprocessing.  

In template_dataset.py,  
    modify_commandline_options:　Add dataset-specific options and rewrite default values for existing options.  
    __init__: Initialize this dataset class.  
    __getitem__: Return a data point and its metadata information.  
    __len__: Return the number of images.  
  
`dataset`    This has our dataset rainy_to_clear(from BDD100k dataset). 

you need to create a data folder with two subdirectories `trainA` and `trainB` that contain images from domain rainy and clear. Create two subdirectories `testA` and `testB` to contain the test set.

`models`    This package contains modules related to objective functions, optimizations, and network architectures. 

In template_model.py,  
    modify_commandline_options:　Add model-specific options and rewrite default values for existing options.  
    __init__: Initialize this model class.  
    set_input: Unpack input data and perform data pre-processing.  
    forward: Run forward pass.This will be called by both optimize_parameters and test.  
    optimize_parameters: Update network weights; it will be called in every training iteration.  
    
`options`    This package options includes option modules: training options, test options, and basic options (used in both training and test).  

`util`    This package includes a miscellaneous collection of useful helper functions.
## Tech used
CycleGAN was proposed in 2017 by Zhu et al. It is a ring network formed by the combination of two mirror-symmetric generative adversarial networks, which are composed of two generators and two discriminators. The training process of CycleGAN is simple and adopts unsupervised learning method. It only needs to use generator and discriminator to complete the transformation of image domain, and then use cyclic consistency to restrict and guarantee the content information of image. Therefore, CycleGAN does not need one-to-one training, and only needs to train two types of images to train a model, which is widely used.  

The following is the formulation of CycleGAN.

![image](https://github.com/foggpoy/Clear-and-Rainy-Transformation-Based-on-CycleGAN-Model/assets/147970661/a715d6db-5e92-48c6-9dde-455873e1853b)  
First we need pictures of two fields, X and Y.In our experiment, for example, if we want to convert the rainy into clear, we need two datasets that are rainy and clear. Here G and F represent two mapping functions, DX and DY represent two discriminators. The goal of G is to convert an image of domain X into an image of domain Y, while F does the opposite. The goal of the discriminators DX and DY is to distinguish between the converted image and the real image.  

In order to ensure the reliability of the transformation, the model introduces two cycle consistency losses. Forward cycle consistency loss ensures that an image converted from X to Y and back to X through F still returns to an image close to the original X. The reverse loop consistency loss ensures that the image converted from Y to X and back to Y is close to the original image Y. In this way, CycleGAN uses these two loss functions to maintain consistency and confidence in the transformation in the absence of paired training data.  

![image](https://github.com/foggpoy/Clear-and-Rainy-Transformation-Based-on-CycleGAN-Model/assets/147970661/01cf7e1c-fb66-4e3e-9974-319e3326640c)

The first is adversarial loss, which is a key component of training generative adversarial network (GAN), where the mapping from domain X to domain Y is considered, which consists of two parts: the generator (G) and the discriminator (DY). And this formula is just a description of how GAN work.

![image](https://github.com/foggpoy/Clear-and-Rainy-Transformation-Based-on-CycleGAN-Model/assets/147970661/3e823076-4e8a-4026-acb6-4b46241d9243)

Ey~pdata(y) and Ex~pdata(x) in the formula represent the expectations for the real data distribution pdata(y) and pdata(x), respectively, where y is the sample from the real data set Y and x is the sample from the real data set X.
The logarithm log here is because in probability theory, we often use logarithmic likelihood to represent probability. Here logDY(y) and log(1-DY(G(x))) represent the logarithmic likelihood of the real image and the generated image, respectively. Therefore, our first term is the expectation of discriminator DY's prediction accuracy for the real Y-domain image, and the second term is the expectation of discriminator DY's prediction accuracy for the generated image. G tries to minimize the formula, and DY tries to maximize it, creating a minimax game. Similarly, there is an adversarial loss function from Y to X.

In addition to the adversarial loss function mentioned above, there is also the key cycle consistency loss function.
![image](https://github.com/foggpoy/Clear-and-Rainy-Transformation-Based-on-CycleGAN-Model/assets/147970661/f0f0144c-02d4-478e-9b87-5d66c9229f45)

The first term of this formula is the difference between the reconstructed image X in x domain mapped to Y domain by G and back to X domain by F and the original image x; the second term is the reverse, for all samples Y from y domain, the difference between the reconstructed image Y mapped to X domain by F and back to y domain by G and the original image Y. The double vertical lines here represent the L1 distance, which is the average of the absolute difference.

So our complete objective function should be the sum of these losses:
![image](https://github.com/foggpoy/Clear-and-Rainy-Transformation-Based-on-CycleGAN-Model/assets/147970661/b7c6bbe9-3c18-4a42-8574-28a3ec09b5a8)

The first and second parts are the adversarial losses of the two GANs we talked about earlier, and the third part is the cycle consistency loss just described, where lamda is a hyperparameter that controls the relative importance of the cycle consistency loss and the adversarial loss.

Optimizing this objective function involves a minimization and maximization process, that is, we try to find the parameters of the mapping functions G and F, and the discriminators DX and DY, such that the entire objective function is minimized, and this process can be expressed in the following form:

![image](https://github.com/foggpoy/Clear-and-Rainy-Transformation-Based-on-CycleGAN-Model/assets/147970661/f9caed47-85a4-461b-982b-c729a0c3bbec)

Here arg min and arg max represent the parameter values that make the given expression reach the minimum and maximum values. This formula means that generators G and F need to maintain cyclic consistency while trying to fool the discriminators DX and DY in order to produce high-quality converted images.
## Installation
The data set used for this project comes from BDD100k. In May 2018, the University of Berkeley AI Lab (BAIR) released BDD100K, the largest and most diverse public driving dataset to date, containing 100,000 high-definition videos at around 40 seconds \720p\30 fps each. The keyframes were sampled at the 10th second of each video to get 100,000 images and annotated.

We chose the part of annotating weather conditions and chose Clear and Rainy as our data sets. The dataset includes a variety of scenarios, lighting conditions, and weather patterns, allowing the model to learn the complex differences between two weather conditions and generalize well to different scenarios.The BDD100k dataset of this experiment contains 5070 rainy day pictures and 37,344 clear pictures in the training set, and 738 rainy day pictures and 5346 clear pictures in the verification set. Partial data and partial sample data sets are selected as shown in the figure below:

Please download our dataset:https://www.kaggle.com/datasets/marquis03/bdd100k-weather-classification
## Bibliography
[1] G. Parmar, T. Park, S. Narasimhan, and J.-Y. Zhu, "One-step image translation with text-to-image models," Mar. 2024. [Online]. Available: https://arxiv.org/pdf/2403.12036.pdf  
[2] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, "Unpaired image-to-image translation using cycle-consistent adversarial networks," in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017.
## Todos

This is an example of our training loss.
![image](https://github.com/foggpoy/Clear-and-Rainy-Transformation-Based-on-CycleGAN-Model/assets/147970661/d3371387-119a-4c1b-90af-470e09520f60)


This is the trained model.
![image](https://github.com/foggpoy/Clear-and-Rainy-Transformation-Based-on-CycleGAN-Model/assets/147970661/c79c1fb6-a2f2-4a68-94ee-1c5c545ff18d)


Here's our rendering.  
`rainy to clear`  

![image](https://github.com/foggpoy/Clear-and-Rainy-Transformation-Based-on-CycleGAN-Model/assets/147970661/d4814415-b84b-4952-9db3-2341e5620e4a)

`clear to rainy`

![image](https://github.com/foggpoy/Clear-and-Rainy-Transformation-Based-on-CycleGAN-Model/assets/147970661/6658265d-b06a-4244-b24b-11ac6df4b708)
