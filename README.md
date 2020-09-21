[![PyPI version](https://badge.fury.io/py/PyALE.svg)](https://badge.fury.io/py/PyALE)
[![Build Status](https://travis-ci.org/DanaJomar/PyALE.svg?branch=master)](https://travis-ci.org/DanaJomar/PyALE)
[![codecov](https://codecov.io/gh/DanaJomar/PyALE/branch/master/graph/badge.svg)](https://codecov.io/gh/DanaJomar/PyALE)

# PyALE

**ALE**: Accumulated Local Effects <br>
A python implementation of the ALE plots based on the implementation of the R package [ALEPlot](https://github.com/cran/ALEPlot/blob/master/R/ALEPlot.R) 

## Installation:
Via pip `pip install PyALE`

## Features:
The end goal is to be able to create the ALE plots whether was the feature numeric or categorical.

### For numeric features:
The package offers the possibility to
* Compute and plot the effect of one numeric feature (1D ALE)
    * including the option to compute a confidence interval of the effect.
* Compute and plot the effect of two numeric features (2D ALE)

### For categorical features:
Since python models work with numeric features only, categorical variables are often encoded by one of two methods, either with integer encoding (when the categories have a natural ordering of some sort e.g., days of the week) or with one-hot-encoding (when the categories do not have ordering e.g., colors)

* For integer encoding: the package offers the option to compute and plot the effect of a discrete feature 
    * including the option to compute a confidence interval of the effect.
* For one-hot-encoding: this part is still [[**under development**]].

## Usage with examples:
* First prepare data and train a model

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# get the raw diamond data (from R's ggplot2) 
dat_diamonds = pd.read_csv('https://raw.githubusercontent.com/tidyverse/ggplot2/master/data-raw/diamonds.csv')
X = dat_diamonds.loc[:, ~dat_diamonds.columns.str.contains('price')].copy()
y = dat_diamonds.loc[:, 'price'].copy()

# convert the three text columns to ordered categoricals
X.loc[:,'cut'] = X.loc[:,'cut'].astype(pd.api.types.CategoricalDtype(
  categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
  ordered=True))
X.loc[:, 'color'] = X.loc[:,'color'].astype(pd.api.types.CategoricalDtype(
  categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J'],
  ordered=True))
X.loc[:, 'clarity'] = X.loc[:, 'clarity'].astype(pd.api.types.CategoricalDtype(
  categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'  ],
  ordered=True))

# use the codes of each categorical as a numeric encoding for the feature
X.loc[:,'cut'] = X.loc[:,'cut'].cat.codes
X.loc[:, 'color'] = X.loc[:, 'color'].cat.codes
X.loc[:, 'clarity'] = X.loc[:, 'clarity'].cat.codes

model = RandomForestRegressor(random_state = 1345)
model.fit(X, y)
```

* import the generic function `ale` from the package

```python
from PyALE import ale
```
* **1D ALE plot for numeric continuous features** 

```python
## 1D - continuous - no CI
ale_eff = ale(
    X=X, 
    model=model,
    feature=['carat'], 
    feature_type='continuous',
    grid_size=50, 
    include_CI=False)
```
![1D ALE Plot](https://raw.githubusercontent.com/DanaJomar/PyALE/master/examples/plots/1D_ALE_Plot_Ex_noCI.jpeg)

The confidence intervals around the estimated effects are specially important when the sample data is small, which is why as an example plot for the confidence intervals we'll take a random sample of the dataset

```python
## 1D - continuous - with 95% CI
random.seed(123)
X_sample = X.loc[random.sample(X.index.to_list(), 1000), :]
ale_eff = ale(
    X=X_sample, 
    model=model,
    feature=['carat'], 
    feature_type='continuous',
    grid_size=50, 
    include_CI=True,
    C=0.95)
```
![1D ALE Plot with CI](https://raw.githubusercontent.com/DanaJomar/PyALE/master/examples/plots/1D_ALE_Plot_Ex_withCI.jpeg)

* **1D ALE plot for numeric discrete features**

```python
## 1D - discrete
ale_eff = ale(
    X=X,
    model=model, 
    feature=['cut'],
    feature_type='discrete')
```
![1D ALE Plot Disc](https://raw.githubusercontent.com/DanaJomar/PyALE/master/examples/plots/1D_ALE_Plot_Discrete_Ex.jpeg)


* **2D ALE plot for numeric features**

```python
## 2D - continuous
ale_eff = ale(
    X=X,
    model=model, 
    feature=['z', 'table'],
    grid_size=100)
```
![2D ALE Plot](https://raw.githubusercontent.com/DanaJomar/PyALE/master/examples/plots/2D_ALE_Plot_Ex.jpeg)

Or sometimes it is better to take a look at the effect of each feature on its own but side by side
For additional plot customization one can pass a figure and axis to the function

```python
## two 1D plots side by side
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
ale_res_1 = ale(
    X=X,  model=model, feature=['z'], feature_type='continuous', grid_size=20, 
    include_CI=True, C=0.95, 
    plot=True, fig=fig, ax=ax1)
ale_res_2 = ale(
    X=X, model=model, feature=['table'], feature_type='continuous', grid_size=20, 
    include_CI=True, C=0.95, 
    plot=True, fig=fig, ax=ax2)
# change x labels
ax1.set_xlabel("depth in mm (0–31.8)")
ax2.set_xlabel("width of top of diamond relative to widest point (43–95)")
```
![1D 2 ALE Plot](https://raw.githubusercontent.com/DanaJomar/PyALE/master/examples/plots/1D_ALE_Plot_2feat_Ex.jpeg)

## Interpretation:

```python
random.seed(123)
X_sample = X.loc[random.sample(X.index.to_list(), 1000), :]
ale_contin = ale(
    X=X_sample, 
    model=model,
    feature=['carat'], 
    feature_type='continuous',
    grid_size=5, 
    include_CI=True,
    C=0.95)
```

![1D ALE Plot](https://raw.githubusercontent.com/DanaJomar/PyALE/master/examples/plots/interpretation_Ex.jpeg)

For continuous variables the algorithm cuts the feature to bins starting from the minimum value and ending with the maximum value of the feature, then computes the average difference in prediction when the value of the feature moves between the edges of each bin, finally returns the centered cumulative sum of these averages (and the confidence interval of the differences - optional). 

```python
ale_contin
```
|  carat  |    eff      | size   | lowerCI_95%   | upperCI_95%    |
| ------  | ----------- | ------ | -----------   | -----------    |
| 0.23    |-1837.001629 |   0.0  |          NaN  |        NaN     |
| 0.38    |-1744.987885 | 215.0  | -1760.904293  |  -1729.071477  |
| 0.56    |-1434.286959 | 189.0  | -1469.568429  |  -1399.005489  |
| 0.90    |  205.224732 | 201.0  |   154.492631  |    255.956834  |
| 1.18    | 1647.159091 | 195.0  |  1491.228257  |   1803.089926  |
| 3.00    | 4637.027675 | 200.0  |  4307.809865  |   4966.245485  |

What interests us when interpreting the results is the difference in the effect between the edges of the bins, in this example one can say that the value of the prediction increases by approximately 2990 (`4637 - 1647`) when the carat increases from 1.18 to 3.00, as can be seen in the last two lines. With this in mind we can see that the values of the confidence interval only makes sense starting from the second value (the upper edge of the first bin) and could also be compared with the eff value of the previous row as to give an idea of how much this difference can fluctuate (e.g., for the bin `(1.18, 3.00]` between 2660 and 3319 with 95% certainty).

The lower edge of the first bin is in fact the minimum value of the feature in the given data, and it belongs to the first bin (unlike the following bins which contain the upper edge but not the lower). The size column contains the number of data points in the bin ending with the feature value in the corresponding row and starting with value before it (e.g., the bin `(1.18, 3.00]` has `200` data points). The rug at the bottom of the generated plot shows the distribution of the data points.

For categoricals or variables with discrete values the interpretation is similar and we also get the average difference in prediction, but instead of bins each value will be replaced once with the value before it and once with the value after it.

```python
ale_discr = ale(
    X=X_sample, 
    model=model,
    feature=['cut'], 
    feature_type='discrete',
    include_CI=True,
    C=0.95)
```

![1D ALE Plot](https://raw.githubusercontent.com/DanaJomar/PyALE/master/examples/plots/interpretation_discr_Ex.jpeg)

We can also think of it from the perspective of bins as follows, every bin contains two consecutive values (or categories) from the feature, for example with the `cut` feature the bins are `[0, 1]`, `[1, 2]`, `[2, 3]`, `[3, 4]`, and what interests us is still the difference in the effect between the edges of the bins, as well as the range in which this difference fluctuate when taking the confidence interval into consideration. That being said the `size` column contains the number of data points in this category (**not** ~~the size of the data sample in the bin~~ anymore), this means to get the sample size in the bin one has to sum up the sample size of each value in it (e.g., in `[0, 1]` there is 38 + 82 data points). The bars in the background of the generated plot shows the size of the sample in each category/value.

```python
ale_discr
```

| cut |        eff  | size  | lowerCI_95%  | upperCI_95% |
| --- | ----------- | ----- | -----------  | ----------- |
| 0   | -97.899068  |   38  |         NaN  |         NaN |
| 1   | -78.637401  |   82  |  -92.021015  |  -65.253787 |
| 2   | -45.933788  |  204  |  -59.335160  |  -32.532416 |
| 3   | -29.779222  |  276  |  -39.234601  |  -20.323843 |
| 4   |  69.394974  |  400  |   51.432901  |   87.357046 |

## Development
* Installing the package in edit mode could be done with `pip install -e`
* `unittest` is used for running the tests 
* `coverage` is used to get the code coverage, which is not an installation requirement of this package, however will be installed if the dev flag was added to pip call i.e., `pip install -e ".[dev]"`
* To get the code coverage report run 
    `coverage run -m --source=PyALE unittest discover -v` then 
    * for a fast report in the shell : `coverage report`
    * for a detailed html report: `coverage html`

* The versions of the packages used during the development process are freezed in requirements.txt
* The latest generated code coverage report could be found in [htmlcov/index.html](https://htmlpreview.github.io/?https://github.com/DanaJomar/PyALE/blob/master/htmlcov/index.html)

### Ref.
* https://cran.r-project.org/web/packages/ALEPlot/vignettes/AccumulatedLocalEffectPlot.pdf

* https://christophm.github.io/interpretable-ml-book/ale.html

  
