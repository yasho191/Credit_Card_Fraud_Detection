<h1 align="center"> Credit Card Fraud Detection </h1>

This Repository contains the official code used for the paper ['Detecting Fraudulent Transactions using Hybrid Fusion Techniques'](https://ieeexplore.ieee.org/document/9664719) - 2021 3rd International Conference on Electrical, Control and Instrumentation Engineering (ICECIE)

For checking out the data science pipleline refer to the Notebook folder and for the main code functions used for creating and evaluation models refer to the Code folder, the dataset can be downloaded from the dataset folder or from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). 

## Abstract of the work

Fraud is one of the most extensive ethical issues in the Financial (Banking) industry. The research aims to create a robust model for predicting fraudulent transactions based on the transactions made by the consumer in the past and present, compare as well as analyse different algorithms that best suit our needs. This paper also focuses on handling the imbalance in the datasets as well as creating a Machine Learning model with high Accuracy, F1-score, AUC, Precision as well as Recall which is achieved using a fusion method in which models are selected from the tested classifiers like Logistic Regression, XGBoost, Random Forest Classifier, Fusion Model, Gaussian NB, and SGDClassifier. Only the models with values of every metric above a certain threshold are selected to churn out maximum performance from the model. The model proposed in this paper uses a probability-based weighted average function for the prediction of fraudulent transactions which yielded a 99% score over all the considered metrics.

### Flowchart of Proposed Method

<img src="Assets/Pipeline.png" />

### Experimental Results

1. Individual Model Performance on Random Oversampling

|      Model Name     |  |             | Random Oversampling        |              |           |
|:-------------------:|:-------------------:|:-----------:|:------:|:------------:|:---------:|
|                     |     F1-Score(%)     | Accuracy(%) | AUC(%) | Precision(%) | Recall(%) |
| Logistic Regression |        92.68        |    92.78    |   97   |     90.96    |   94.57   |
|    SGD Classifier   |        90.22        |    89.95    |   90   |     91.65    |   89.17   |
|     Gaussian NB     |        86.76        |    87.98    |   95   |     78.75    |   96.59   |
|    Random Forest    |        97.77        |    97.14    |   99   |      100     |   96.34   |
|       XGBoost       |        98.59        |    98.47    |   99   |      100     |   97.38   |
      
2. Individual Model Performance on SMOTE Oversampling

|      Model Name     |  |             | SMOTE Oversampling       |              |           |
|:-------------------:|:------------------:|:-----------:|:------:|:------------:|:---------:|
|                     |     F1-Score(%)    | Accuracy(%) | AUC(%) | Precision(%) | Recall(%) |
| Logistic Regression |        96.52       |    96.52    |   99   |     96.03    |   97.06   |
|    SGD Classifier   |        95.31       |    95.23    |   95   |     96.07    |   94.63   |
|     Gaussian NB     |        88.13       |    89.19    |   98   |     80.28    |   97.68   |
|    Random Forest    |        97.61       |    96.99    |   99   |     99.99    |   96.03   |
|       XGBoost       |        97.38       |    97.03    |   99   |     99.99    |   95.29   |


3. Hybrid Model Performance on Random and SMOTE Oversampling

While creating the hybrid model a probability based weighted average function has been considered, this function takes into consideration the probailty of the prediction of each model and their respective f1-score to calculate the final prediction.

|  Sampling Technique | F1-Score(%) | Accuracy(%) | AUC(%) | Precision(%) | Recall(%) |
|:-------------------:|:-----------:|:-----------:|:------:|:------------:|:---------:|
| Random Oversampling |    99.99    |    99.99    |  99.99 |      100     |   99.99   |
|  SMOTE Oversampling |    99.73    |    99.73    |  99.73 |     99.66    |   99.79   |

4. Confusion Matrix for Hybrid Models

<img src="Assets/ConfusionMatrix.png" height=350 width=800/>

### Citation

```py
Authors: Y. Shinde, A. S. Chadha and A. Shitole, 
Paper Title: "Detecting Fraudulent Transactions using Hybrid Fusion Techniques," 
Conference Name: 2021 3rd International Conference on Electrical, Control and Instrumentation Engineering (ICECIE), 
Year: 2021, pp. 1-10, 
DOI: doi: 10.1109/ICECIE52348.2021.9664719.
```
