# Credit_Risk_Analysis

## Overview:
Use of credit card dataset from LendingClub lending company to evaluate machine learning models in loan risk prediction.  Credict risk deals with unbalanced risky loans (good loans are the majority).

## Purpose:
Reduce bias of under/over group representation in order to evaluate model performance that would best predict credit risk. As a result, to predict if a loan application is worthy of approval or not.

## Methodology:
### Tools used:
- Naive Random Oversampler
- SMOTE Oversampler
- Cluster Centroids Undersampler
- SMOTEENN
- Balanced Rando Forest Classifier
- Easy Ensample AdaBoost Classifier

### Data information and handling:
- Data represents a loan application approval (high risk vs low risk)
- High risk is greatly undersampled (N = 347) vs low risk (N = 684700
- Data is separated into features(X) and target (y)
- Data values are converted into numerical ones with pd.get_dummies()

## Results:
## Deliverable 1
## Oversampling
### Naive Random Oversampler:
(instances are randomly selected and added to the minority class)
- Calculated accuracy_score is low (0.662) which means that the **Naive Random Oversampler** has only 66% accuracy.  To be considered good enough at predicting it should be at least 70% or higher.

IMAGE 1

- Classification Report: for "high risk" precision is very low (0.01) and recall is at acceptable level (0.72).  This means that there is a good amount of false positives (most of its predicted values are incorrect).  For "low risk", precision is very high (1.0) and recall is 0.60. This high precision score indicates that the model returns accurate results but only at 66% accuracy (accuracy_score).

IMAGE 2 

### SMOTE: 
(for an instance from the minority group, a number of its closest neighors is selected to create new instances)
- Calculated **SMOTE**accuracy_score is low (0.656). It means that it's only 66% good at predicting accuracy.

IMAGE 3

- Classification Report: for "high risk" precision is very low (0.01) and recall is at acceptable level (0.61).  This means that there is a good amount of predicted values that are incorrect).  For "low risk", precision is ideal at (1.0) and recall is 0.70. High scores for both show that the model is returning accurate results; however, with quite low accuracy_score of 66%.

IMAGE 4

## Undersampling
### Cluster Centroids 
(the opposite approach of oversampling)
