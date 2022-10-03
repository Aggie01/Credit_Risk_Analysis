# Credit_Risk_Analysis

## Overview
Use of credit card dataset from LendingClub lending company to evaluate machine learning models in loan risk prediction.  Credict risk deals with unbalanced risky loans (good loans are the majority).

## Purpose
Reduce bias of under/over group representation in order to evaluate model performance that would best predict credit risk. As a result, to predict if a loan application is worthy of approval or not.

## Methodology
### Tools used:
- Naive Random Oversampler
- SMOTE
- SMOTEENN
- Balanced Rando Forest Classifier
- AdaBoost Classifier
- Easy Ensample Classifier
### Data information and handling:
- Data represents a loan application approval (high risk vs low risk)
- High risk is greatly undersampled (N = 347) vs low risk (N = 684700
- Data is separated into features(X) and target (y)
- Data values are converted into numerical ones with pd.get_dummies()

## Results
## Oversampling
### Naive Random Oversampler:
- Calculated accuracy_score is low (0.662) which means that Naive Random Oversampler is only 66% good at accurate .  To be considered good enough at predicting it should be at least 70% or higher

IMAGE 1

- 
