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
