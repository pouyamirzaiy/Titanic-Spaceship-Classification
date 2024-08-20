# **Titanic Spaceship Dataset Analysis and Binary Classification**

## **Project Overview**

This project involves analyzing the Titanic Spaceship dataset and performing binary classification to predict whether passengers were transported or not. The project includes exploratory data analysis, data cleaning, feature engineering, model training, and evaluation.

## **Libraries and Dependencies**

The following libraries are used in this project:

- **Warnings**: Handling warnings
- **NumPy**: Linear algebra
- **Pandas**: Data processing
- **Matplotlib**: Visualization
- **Seaborn**: Visualization
- **Optuna**: Hyperparameter optimization
- **Scikit-learn**: Preprocessing, model training, and evaluation
- **XGBoost**: Extreme Gradient Boosting

You can install the required libraries using the following command:

```bash
pip install numpy pandas matplotlib seaborn optuna scikit-learn xgboost
```

## **Data Files**

- `train.csv`: Training dataset
- `test.csv`: Test dataset

## **Project Structure**

- `src/`
  - `data_cleaning.py`: Script for data cleaning and preprocessing.
  - `feature_engineering.py`: Script for feature engineering.
  - `model_training.py`: Script for training and evaluating models.
  - `visualizations.py`: Script for creating visualizations.
  - `ablation_study.py`: Script for performing ablation study.
- `notebooks/`
  - `Titanic.ipynb`: Jupyter notebook for exploratory data analysis and model evaluation.
- `README.md`: Project overview and instructions.

## **Data Exploration**

The data exploration phase includes examining the distributions of numerical and categorical variables, visualizing correlations, and identifying missing values. Detailed analysis can be found in the `Titanic.ipynb` notebook.

## **Data Cleaning**

The data cleaning phase includes handling missing values and transforming columns for better analysis. Detailed steps can be found in the `data_cleaning.py` script.

## **Feature Engineering**

Feature engineering involves encoding categorical variables into numeric values and creating new features. Detailed steps can be found in the `feature_engineering.py` script.

## **Model Training and Evaluation**

Various machine learning models are trained and evaluated using different metrics. The models include:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- XGBoost Classifier

Detailed steps can be found in the `model_training.py` script.

## **Ablation Study**

An ablation study is performed to evaluate the impact of different features on the model's performance. Detailed steps can be found in the `ablation_study.py` script.

## **Results**

The results of the model evaluation and ablation study can be found in the `Titanic.ipynb` notebook.

## **Contact Information**

If you have any questions or feedback, feel free to reach out to me at pouya.8226@gmail.come
