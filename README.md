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
- High risk is greatly undersampled (N = 347) vs low risk (N = 684700)

![](image%200.jpg)

- Data is separated into features(X) and target (y)
- Data values are converted into numerical ones with pd.get_dummies()

## Results:
## Deliverable 1
## Oversampling
### Naive Random Oversampler:
(instances are randomly selected and added to the minority class)
- Calculated accuracy_score is low (0.662) which means that the **Naive Random Oversampler** has only 66% accuracy.  To be considered good enough at predicting it should be at least 70% or higher.

![](image%201.jpg)

- Classification Report: for "high risk" precision is very low (0.01) and recall is at acceptable level (0.72).  This means that there is a good amount of false positives (most of its predicted values are incorrect).  For "low risk", precision is very high (1.0) and recall is 0.60. This high precision score indicates that the model returns accurate results but only at 66% accuracy (accuracy_score).

![](image%202.jpg)

### SMOTE: 
(for an instance from the minority group, a number of its closest neighors is selected to create new instances)
- Calculated **SMOTE** accuracy_score is low (0.656). It means that it's only 66% good at predicting accuracy.

![](image%203.jpg)

- Classification Report: for "high risk" precision is very low (0.01) and recall is low as well (0.61).  This means that there is a good amount of predicted values that are incorrect).  For "low risk", precision is ideal at (1.0) and recall is at acceptable level of 0.70. High scores for both show that the model is returning accurate results; however, with quite low accuracy_score of 66%.

![](image%204.jpg)

## Undersampling
### Cluster Centroids: 
(the opposite approach of oversampling. The majority class is undersampled down to the size of the minority)
- Calculated **Cluster Centroids** accuracy_score is low (0.656). It means that it's only 66% good at predicting accuracy

 ![](image%205.jpg)
 
 - Classification Report: very low precision for "high risk" (0.01).  Recall is also at low level (0.69).  This means that most of its predicted instances are incorrect when compared to the training data.  For "low risk", precision is at (1.0) and recall at a very low level (0.40). Such low recall level relates to high level of false negatives
 
![](image%206.jpg)

## Deliverable 2
### Combination Sampling
(it combines both sampling methods, over and under sampling. The minority group is oversampled but also each group's outliers are removed)
- Calculated **Combination Sampling** accuracy_score is low (0.544). It means that the Combination Sampling is only 54%  accurate

![](image%207.jpg)

- Classification Report: "high risk" precision is very low as compared to "low risk."  It means that there are many false positives in "high risk."  Recall is at low level for "low risk."  

![](image%208.jpg)

## Deliverable 3
##  Ensemble Learners
### Balanced Random Forest Classifier:
(builds random subset of features)
- Calculated **Random Forest Classifier** accuracy_score is quite high (0.788). It means that the Balanced Random Forest Classifier is 79% accurate

![](image%209.jpg)

- Classification Report: the results show the precision for "high risk" loans is very low, indicating a large number of false positives. Recall is at a moderate level which indicates that average number of false negatives. Overall, Random Forest is fine because its accuracy is at 79%.

![](image%2010.jpg)

### AdaBoost Classifier:
(it increases the weight of "weak" group)
- Calculated **AdaBoost Classifier** accuracy_score is very high, 93%. 

![](/image%2011.jpg)

- Classification Report: very high score for "low risk" for both, precision and recall.  High scores for both indicate that the classifier is returning accurate results (high precision), as well as it is returning a majority of all positive results (high recall).  For "high risk," there is high recall but low precision which means that most results that are returned are incorrect when compared to training set.

![](image%2012.jpg)

## Summary:
- Overall, oversampling did not produce accurate and thrustworthy predictions.  Accuracy scores were below 70%. 
- The quality of a positive prediction made by the Naive Random Sampling for "low risk" group is at an ideal level, 100%.  It means that most of its predicted instances are correct when compared to the training ones; however, at low accuracy.
- SMOTE oversampler produced low scores for "high risk" in both, precision and recall. It implies a poor prediction.
- Overall, undersampling generated low acceptance score, below 70%. Also, low overall recall shows high level of false negatives.
- Combination Sampling shows very poor accuracy, only 54%.
- Ensemble Learners generated overall winning accuracy results, Random Forest 79% and AdaBoost impressive 93%.  AdaBoost produced very high on "low risk" group for both precision and recall.  For that reason it is recommended that AdaBoost is applied as it is successful at producing the most accurate results.
- 
- 












