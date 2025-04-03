# Rainfall Prediction with XGBoost

This project is for the Kaggle Playground Series S5E3: [Binary Prediction with a Rainfall Dataset](https://www.kaggle.com/competitions/playground-series-s5e3/overview).

In cross-validation on training data, the resulting model shows an 83.7% accuracy in predicting rainfall and a 0.891 AUC of predicted probabilities.

In competition evaluation, the model showed an increased 0.8963 AUC of predicted probabilities. This placed in the 80th percentile of competitors (#847 out of 4,382 entrants). A previous version of this analysis without feature engineering would have placed in the 85th percentile (#640 out of 4,382 based on an AUC of 0.8975).

## Competition Background

Per Kaggle:

- The dataset for this competition (both train and test) was generated from a deep learning model trained on the [Rainfall Prediction using Machine Learning](https://www.kaggle.com/datasets/subho117/rainfall-prediction-using-machine-learning) dataset.
- The goal is to predict rainfall for each day of the year.
- Submissions are evaluated on [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probability and the observed target.

## Dataset

Project data is split into training and testing datasets.

- [train.csv](data/train.csv) contains 2,190 rows of data to train and validate the model.
- [test.csv](data/test.csv) contains 730 rows of data to score submissions.
- [sample_submission,csv](data/sample_submission.csv) shows the format for submitted predictions—a probability of rain should be predicted, not a binary outcome.

Training and testing data contains 11 input features:

- `day` | integer number of day in the dataset, sequential starting with 1
- `pressure` | float value of atmospheric pressure in millibars
- `maxtemp` | float, maximum temperature during day in degrees Celsius
- `temparature` (sic) | float, average temperature during day in degrees Celsius
- `mintemp` | float, minimum temperature during day in degrees Celsius
- `dewpoint` | float, temperature at which dew forms (depending on pressure and humidity) in degrees Celsius
- `humidity` | float, Relative Humidity (percentage out of 100)
- `cloud` | float, percentage of cloud cover
- `sunshine` | float, percentage of time with direct sunshine

Training data also include `rainfall`, a binary outcome label of whether there was rainfall on that day.

## Analysis

All analysis was done in a Jupyter Notebook [file](rainfall.ipynb). Because my focus was on model building, exploratory data analysis was abbreviated, and pre-processing was aided with public resources.

- Public [notebooks](https://www.kaggle.com/competitions/playground-series-s5e3/code) for the competition were referenced to help speed up understanding of the data.
- Credit to Lekha Topil for their notebook [Rainfall Prediction | EDA | LightGBM | CatBoost](https://www.kaggle.com/code/lekhatopil/rainfall-prediction-eda-lightgbm-catboost) and its feature engineering functions. This created a corrected `day` column, a `month` column, and several interaction features.
- Data pre-processing and feature engineering could also be abbreviated because I intended to use an XGBoost model, which has methods for regularization and scaling feature importance.

The focus of my model was practicing optimization of an XGBoost model with hyperparameter tuning. Please refer to the [XGBoost Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html) documentation for more information on all the available parameters. The primary scoring metric was AUC because it's the competition evaluation metric, but I also tracked overall Accuracy.

- I created a basic XGBoost model with mostly default parameters to assess baseline performance — 86.3% Acc, 0.885 AUC.
- I performed a grid search using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) for several XGBoost parameters: `max_depth`, `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`, and `reg_alpha`.
  - The primary goal of these parameters is to prevent the model from overfitting the current data and having worse performance on new data, which was especially important with a small dataset.
  - I initially performed the cross-validation grid searches with wider ranges for possible parameter values, but I re-ran CVGridSearch with smaller search spaces to fine-tune the values and cut down on computation time.
  - The best cross-validation score was 0.885 AUC, not a significant improvement from the base model.
- I then used the optimal hyperparameters to perform a second grid search for `n_estimators` and `learning_rate`, the two most powerful parameters.
  - Like the first grid search, I refined the search space for subsequent runs to cut down computation and fine-tune values.
  - The best cross-validation score was 0.881, which was a slight decrease from the base model. However, this score was calculated through 5-fold cross-validation which is a more conservative evaluation than a train-test split.

I fit a final model with the optimal parameters decided by the grid searches. The model performed similarly on the training and testing splits (0.910 vs 0.899 AUC), which would suggest that the model wasn't overfitting. With a lower chance of overfitting, I re-fit the model on all of the available training data.

- I performed a 5-fold cross validation for an accurate estimation of model performance.
- Average accuracy was 83.7%, and average AUC was 0.891.

## Results and Findings

The final model showed an 83.7% accuracy and an AUC of 0.891 based on the predicted probabilities. The accuracy decreased slightly from the base model, but the AUC increased slightly.

In competition evaluation, the model showed an increased 0.8963 AUC of predicted probabilities. This placed in the 80th percentile of competitors (#847 out of 4,382 entrants).

A few findings:

- Hyperparameter tuning was only able to squeeze out very minor improvements in model performance, even with full grid searches of large ranges for many parameters.
- The model performed worse than the base model in accuracy but higher in AUC.
  - Hyperparameter tuning via grid search is going to push towards "optimal" performance but only for a given metric.
  - It's up to interpretation whether this trade-off is worthwhile. In this case, it makes sense to optimize the metric for competition scoring over any other trade-off.
  - If multiple metrics are critical, GridSearchCV does accept custom scoring functions. A custom scoring function could balance multiple criteria.

## Takeaways

Hyperparameter tuning can offer meaningful improvements in model performance. Grid searching for optimal hyperparameters can be computationally inefficient, but they are feasible on smaller datasets.

Some ideas for future improvements with more time:

- On larger datasets, [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) would be significantly more efficient than GridSearchCV.
- Other tree-based models should be evaluated, specifically [LightGBM](https://lightgbm.readthedocs.io/en/stable/) (especially on larger datasets) and [CatBoost](https://catboost.ai/).
- Additional feature engineering could have greater performance improvements than better hyperparameter tuning—here and generally.
  - This is challenged by the results of the competition, where my initial model—without any additional feature engineering—would have scored better than the model after feature engineering (0.89748 versus 0.89626 AUC).
  - This would suggest that adding too many additional features to a small dataset can lead to overfitting, even with XGBoost hyperparameters designed to regulate feature importance.
- Based on competition scoring, this model is performing worse on the test data, so additional steps could be taken to prevent overfitting.
  - Ensembling multiple models could produce a more robust model.
  - Pruning features could reduce overfitting—after feature engineering, there are 61 inputs.
  - Earlier versions of the model without any additional feature engineering performed better.

As an exercise in hyperparameter tuning for XGBoost, this analysis was a strong success.
