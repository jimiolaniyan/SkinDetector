# Skin Detector
Python application that detects parts of an image that contain skin. It uses the [UCI Skin Segmentation](https://archive.ics.uci.edu/ml/datasets/skin+segmentation) data set to train a basic generative classifier which decides whether a pixel is skin or non-skin.

## Results
With a threshold of `Pr(x|θ) > 0.7`, the test set produced 92% accuracy. The results of the classifier on random images are shown below:

...

### Dependencies
```
pandas      0.22.0
numpy       1.14.3
scip        1.1.0
opencv3     3.2.0
```
 
