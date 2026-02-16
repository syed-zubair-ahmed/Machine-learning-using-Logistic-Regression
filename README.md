---

# Machine Learning using Logistic Regression

This project demonstrates how to build a **Logistic Regression classification model** using the Titanic dataset. The goal is to predict whether a passenger survived or not based on various features.

---

## Project Overview

Logistic Regression is a supervised machine learning algorithm used for **classification problems**. Unlike Linear Regression (which predicts continuous values), Logistic Regression predicts probabilities and classifies data into discrete categories.

In this project:

* We use the **Titanic dataset**
* Perform data preprocessing
* Train a Logistic Regression model
* Evaluate performance using multiple metrics
* Visualize results using a confusion matrix
* Save the trained model using pickle

---

##  What is Logistic Regression?

Logistic Regression:

* Outputs probabilities using the **Sigmoid function**
* Is mainly used for:

  * Binary Classification
  * Multi-class Classification
  * One-vs-Rest Classification

###  Assumptions:

* Dependent variable must be categorical
* Independent variables should not be highly correlated
* No extreme outliers
* Features should be independent

---

## Dataset Used

We used the built-in **Titanic dataset** from Seaborn.

Target Variable:

* `survived` → 0 (Did not survive), 1 (Survived)

Features include:

* Passenger class
* Gender
* Age
* Fare
* Family information
* Embarkation details
* And more

---

##  Libraries Used

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
pickle
```

---

## Project Workflow

### 1️ Data Loading

```python
df = sns.load_dataset('titanic')
```

---

### 2️ Data Preprocessing

* Dropped `deck` column (too many missing values)
* Filled missing values in:

  * `age` → median
  * `fare` → median
  * `embarked`, `embark_town` → mode
* Encoded categorical features using `LabelEncoder`

---

### 3️ Train-Test Split

```python
train_test_split(test_size=0.2, random_state=42)
```

* 80% training
* 20% testing

---

### 4️ Model Training

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

---

### 5️ Model Evaluation

Metrics used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

### Model Performance

```
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1 Score: 1.0
```

Confusion Matrix:

```
[[105   0]
 [  0  74]]
```

---

## Confusion Matrix Visualization

A heatmap visualization was created using Seaborn to analyze prediction performance.

---

## Model Saving

The trained model is saved using pickle:

```python
pickle.dump(model, open('./saved_models/02_model_logistic_regression.pkl', 'wb'))
```

This allows reuse of the model without retraining.

---

## Project Structure

```
Machine-learning-using-Logistic-Regression/
│
├── logistic_regression.ipynb
├── saved_models/
│   └── 02_model_logistic_regression.pkl
├── README.md
```

---
