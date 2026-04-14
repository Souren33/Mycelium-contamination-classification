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
- [Lit Review](#lit-review)
- [Final Report](#final-report)
- [Project Authors](#project-authors)


## About
This project tests the use of two machine learning methods for the quick identification of contamination within fungi. Much of the current methodology for identifying contamination carries significant technological requirements, as noted in our [Lit Review](#lit-review) below. We hope that applying our two methods: **Bayesian** and **CNN** will create an opportunity for consumer-level or grassroots groups to achieve the same or similar results on a budget.

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
TBD


### Bayesian
TBD


## Lit Review
Our review of current application of fungi classification using ML methods can be found <a href="https://docs.google.com/document/d/1nX6VJCRoBd36EaXLEqsjAtA8lDCCTGIvGkf0JazuqYY/edit?usp=sharing">here</a>
                                                                                                                                                                       
## Final Report
Our final report can be found...

## Project Authors
Kevin Martone | martone.k@northeastern.edu | [Github](https://github.com/kevinmartone)

Souren Prakash | prakash.so@northeastern.edu | [Github](https://github.com/Souren33)
