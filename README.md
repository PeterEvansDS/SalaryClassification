# Salary Classification
Model to predict whether someone receives a salary of over 50k using their characteristics listed in the census, using the dataset from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Adult).

Project contains:
- Data cleaning (including categorical encoding) and exploration (`prep.py`)
- Classifiers built with K-nearest Neighbours, Logisitic Regression, Decision Tree, Random Forest, Support Vector Classifier, and Naïve Bayes algorithms (`build.py`)
- Evaluation by accuracy and F-score
- Improvement to models through feature selection and hyperparameter tuning using Scikit-learn's RFECV and SequentialFeatureSelection for the former and GridSearchCV for the latter.
- Investigation into the combination of models using Voting and Stacking ensemble methods.

Description of all processes is written up in the [report](https://github.com/PeterEvansDS/SalaryClassification/blob/main/report.pdf).

![alt text](https://github.com/PeterEvansDS/SalaryClassification/blob/main/images/money.jpeg)

