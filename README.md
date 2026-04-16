<div align="center">
  <h1>🍄 ML Mycelium Classification 🍄</h1>
  <p>Detecting fungi contamination using Bayesian and CNN methods</p>
</div>

---

### Kevin Martone and Souren Prakash

## Table of Contents
- [About](#about)
- [Set Up Instructions](#set-up-instructions)
- [Code Breakdown](#code-breakdown)
- [Final Report](#final-report)
- [Project Authors](#project-authors)


## About
This project tests the use of two machine learning methods for the quick identification of contamination within fungi. Much of the current methodology for identifying contamination carries significant technological requirements, as noted in our [Final Report](#final-report) below. We hope that applying our two methods: **Bayesian** and **CNN** will create an opportunity for consumer-level or grassroots groups to achieve the same or similar results on a budget.

## Set Up Instructions

- The necessary requirements can be found via:
```
python_requirements.txt
r_requirements.r
```


## Code Breakdown
The models we plan to utilize are Bayesian and CNN, with the CNN being implemented with python and the bayesian model through R.

### Preprocessing
Both our models require preprocessing of the input data. Our data is sourced from Kaggle <a href="https://www.kaggle.com/datasets/oasisdata/mycelium-contamination-images">here</a>.
The dataset is composed images of mycelium in various stages labelled by being either clean or contaminated. For our purposes with both the Bayesian model and the CNN model we want uniform dimensions for our images. Python was utilized transform all images to (128 x 128) while maintaing RGB channels as they represent an important feature for our classification.

Further adjustments are only required for the Bayesian model. In order generate our features we need to convert image data into numerical data. This is done within python, and the first stage involves converting each image to one row of a dataframe. Each column of said dataframe will be a unique (r/g/b)_pixel#. Resulting in a final df with 500 rows and 2 + 3*(128^2) columns (adding two for filename and label). Further modification to the data is done to speed up the Bayesian model, but you can read about that further within our [Final Report](#final-report)

### CNN
Two CNN variants were developed using Python and TensorFlow Keras.
#### Custom Keras Model

- 3×3 kernel size across all convolutional layers
- ReLU activation throughout, sigmoid output for binary classification
- Max pooling with 2×2 kernel
- Initial performance was poor, leaky ReLU provided only marginal improvement

#### ResNet50 (Transfer Learning)
After identifying that color and texture are the key visual indicators of contamination, a pretrained ResNet50 backbone was used:
- ResNet50 weights pretrained on ImageNet loaded as the feature extractor
- Custom classification head added for binary output
- Fine-tuned on the mycelium dataset
- Performance improved substantially over the custom Keras model

#### Validation
Stratified k-fold cross-validation was applied:
- Dataset size (~500 images) made a standard train/test split unreliable
- Stratified sampling ensured class balance was maintained across folds
- K-fold allowed the full dataset to contribute to both training and evaluation given the limited data available

#### Note on Comparison with Bayesian Model
The CNN models use k-fold cross-validation while the Bayesian model uses a standard 80/20 train/test split. This means the two approaches are not evaluated on identical holdouts and direct metric comparison should be interpreted with that caveat in mind.


### Bayesian

The Bayesian model was implemented from scratch within R and uses a metropolis-hastings MCMC approach for sampling. the 49,152 features are compressed to 24 color histograms/bins, 8 per color. Each channel is normalized independently so each color sums to 1.
- Total red pixels = x
- red pixels in the 0 - 0.125 bin = y
- normalized value = y/x

#### Priors
As the values fall between (0,1) we will be using a beta prior to estimate mu, and gamma for tau (1/variation(sigma^2). Because these priors are non-conjugate, the use of an MCMC sampling approach was required.

#### Sampling
independently per feature per class (48 chains total):
- 2000 iterations, 500 burn-in discarded, thinning by 5
-   Initial attempts used burn-in of 200, and thinning of 2. This resulted in higher autocorrelation between samples and worse chaining.
-   Decreasing our proposed step size of mu from .05 to .01 as our data is rather narrow already given it has been twice normalized.

#### Predicting

Log predictive likelihood is done Monte Carlo integration.
Rather than plugging in single point estimates for mu and tau, we average the Gaussian likelihood across all posterior samples per feature.

Log predictive likelihoods are summed across all 24 features and combined with the log class prior. The class with the higher combined score wins via argmax.

                                                                                                                                                                     
## Final Report
Our final report can be found...

## Project Authors
Kevin Martone | martone.k@northeastern.edu | [Github](https://github.com/kevinmartone)

Souren Prakash | prakash.so@northeastern.edu | [Github](https://github.com/Souren33)
