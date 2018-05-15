# Skin Detector
Python application that detects parts of an image that contain skin. It uses the [UCI Skin Segmentation](https://archive.ics.uci.edu/ml/datasets/skin+segmentation) data set to train a basic generative classifier which decides whether a pixel is skin or non-skin.

## Results
With a threshold of `Pr(x = 1|θ) > 0.7`, the test set produced 92% accuracy. The results of the classifier on random images are shown below:

![test](https://raw.githubusercontent.com/jimiolaniyan/SkinDetector/master/images/original/test.jpeg)  ![mask](https://raw.githubusercontent.com/jimiolaniyan/SkinDetector/master/images/mask/mask.jpg)\
![test2](https://raw.githubusercontent.com/jimiolaniyan/SkinDetector/master/images/original/test1.jpeg)   ![mask2](https://raw.githubusercontent.com/jimiolaniyan/SkinDetector/master/images/mask/mask1.jpg)\
![test3](https://raw.githubusercontent.com/jimiolaniyan/SkinDetector/master/images/original/test5.jpeg)   ![mask3](https://raw.githubusercontent.com/jimiolaniyan/SkinDetector/master/images/mask/mask5.jpg)\
![test4](https://raw.githubusercontent.com/jimiolaniyan/SkinDetector/master/images/original/test6.jpg)   ![mask3](https://raw.githubusercontent.com/jimiolaniyan/SkinDetector/master/images/mask/mask6.jpg)

## Observations
1. It is clear that the generative classifier approach is able to separate skin pixels in certain regions with a good level of confidence. However, it also suffers from creating false positives, as it is evident in the first image where the it classifies the shirt of the individuals at the extremes of the photo as skin pixels.
2. Another observation that is worth mentioning is that the classifier does not classify dark skin regions with high confidence. This can be observed in first image (face of man at extreme left) and the last image (hand at top left). Although the classifier is not perfect, it is possible that there is some bias in the training data set towards lighter skin pixels. i.e. dark skin pixels are not well represented in the UCI Skin Segmentation data set.     

### Dependencies

> * pandas      0.22.0
> * numpy       1.14.3
> * scipy       1.1.0
> * opencv3     3.2.0
