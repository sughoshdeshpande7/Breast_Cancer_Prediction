# Breast Cancer Predictor: Project Overview
* Built a model that accepts cell nucleus values features of a breast cancer tumor as input and predicts if the cancer is Benign or Malignant.
* Model is trained on a dataset of 570 Breast Cancer Images.
* Data was trained on 5 different models. K-fold cross-validation was performed to validate for overfitting and a final trained Support Vector Machine (SVM) model was used to build the predictor.

## Code and Resources used  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, selenium, pickle  
**For Web Framework Requirements:**  ```pip install -r requirements.txt```  
**Dataset Used:**  https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data<br>
**K-Fold Cross Validation Github:** https://github.com/Inyrkz/breast_cancer/blob/main/k_fold_cv_article_guide.ipynb <br>
**K-Fold Cross Validation Article:** https://www.section.io/engineering-education/how-to-implement-k-fold-cross-validation/<br>
**YouTube Reference:** https://www.youtube.com/watch?v=NSSOyhJBmWY

## Data Cleaning
Following changes were made to the data to make it usable for a model:
*	Column with Null Values was removed.
*	Got the count of malignant vs benign tumor cells.
*	Performed encoding to to represent categorical variables as numerical values to use it in the ML model.

## EDA
Various analysis was made related to the dataset and the models. Below are a few highlights. 

![alt text](https://github.com/sughoshdeshpande7/Breast_Cancer_Prediction/blob/ba700fc9bb3c888ee818969885db370f3e67236e/Images/tumors.png)
![alt text](https://github.com/sughoshdeshpande7/Breast_Cancer_Prediction/blob/ba700fc9bb3c888ee818969885db370f3e67236e/Images/models.png)

## Model Building 

StandardScaler method was used to remove the mean and scale each feature/variable to unit variance. The data was split into train and test sets with a test size of 20%.  
Five different models were tried and evaluated based on their metrics:
*	**Logistic Regression** 
*	**K-Nearest Neighbor**
*	**Decision Tree Classifier** 
*	**Random Forest Method**
*	**Support Vector Machines**

## Model performance
The SVM model outperformed the other approaches on the test and validation sets. 
*	**Random Forest** : Accuracy = 94.73%
*	**Decision Tree** : Accuracy = 93.85%
*	**Logistic Regression**: Accuracy = 96.49%
*	**K-NN**: Accuracy = 95.61%
*	**SVM**: Accuracy = 98.24%

## Deployment
A Final Trained model was built on SVM where the input of the nucleus features are accepted from the user and the model predicts if the tumor is malignant or benign. The Final model can be downloaded from ```svm_model.pkl```

  
