# Decision Trees and Ensemble Models

In this assignment, we explore decision trees and their ensemble counterparts using the **Breast Cancer Dataset**. The tasks focus on building classifiers, evaluating their performance, analyzing feature importance, and optimizing hyperparameters. The assignment is divided into six key sections.

---

## Objectives
1. Understand and implement decision trees for classification tasks.
2. Explore and evaluate ensemble models like Random Forest and XGBoost.
3. Analyze feature importance for better insights into the dataset.
4. Utilize hyperparameter tuning to optimize model performance.

---

## Dataset
The **Breast Cancer Dataset** from Scikit-learn is used in this assignment. It includes:
- **Features**: Numerical data representing various attributes of breast cancer cases.
- **Target**: Binary classification labels indicating the diagnosis (Malignant or Benign).

---

## Tasks Overview

### 1. Data Loading and Exploration
- Load the Breast Cancer Dataset using `load_breast_cancer` from Scikit-learn.
- Assign features to `X` and target labels to `y`.

### 2. Data Preprocessing
- Split the data into training and testing sets using `train_test_split` from Scikit-learn.
- Use a suitable `test_size` and `random_state` for reproducibility.

### 3. Decision Trees
- Train a **DecisionTreeClassifier** on the training data.
- Predict and evaluate the classifier's accuracy on the test data.
- Visualize the decision tree using `plot_tree` with feature names for clarity.

### 4. Random Forest
#### a. Random Forest Implementation
- Train a **RandomForestClassifier** and evaluate its accuracy on the test data.
- Predict and evaluate the classifier's accuracy.

#### b. Feature Importance Analysis
- Analyze feature importance using the `feature_importances_` attribute of the Random Forest model.
- Sort and display the features in descending order of importance.

### 5. XGBoost
- Train a classifier using `xgboost.XGBClassifier`.
- Evaluate its accuracy on the test data.

### 6. Hyperparameter Tuning
- Apply **Grid Search** with cross-validation to optimize hyperparameters for the Random Forest model.
- Evaluate the optimized model's accuracy on the test set.

---

## Key Libraries and Tools
The following libraries are used in this assignment:
- **NumPy**: For numerical operations and data manipulation.
- **Matplotlib**: For visualizing the decision tree.
- **Scikit-learn**: For machine learning models, data preprocessing, and evaluation metrics.
- **XGBoost**: For gradient-boosted tree algorithms.

---
