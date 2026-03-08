# 🩺 Diabetes Prediction Project

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)  
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-orange)](https://scikit-learn.org/)  
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-green)](https://xgboost.readthedocs.io/)  
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🔹 Project Overview
This project uses **machine learning** to predict the **onset of diabetes** using diagnostic health metrics.  
The main goal is to **maximize recall**—catching as many positive cases as possible—while keeping predictions reliable.

Key objectives:  
- Identify patients at risk of diabetes early  
- Build an interpretable and robust ML pipeline  
- Provide a clear workflow for preprocessing, feature engineering, and evaluation

---

## 📊 Dataset Description
The dataset `diabetes.csv` contains **diagnostic measurements** and an `Outcome` column:

| Feature | Description |
|---------|-------------|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2-hour oral glucose tolerance test)|
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | Genetic risk score |
| `Age` | Age (years) |
| `Outcome` | 0 = Non-diabetic, 1 = Diabetic |

---

## 🧩 Project Workflow

### 1️⃣ Exploratory Data Analysis (EDA)
- Checked dataset shape, statistics, and missing values  
- Visualized distributions of features with histograms and boxplots  
- Studied correlations between features and `Outcome`  
- Identified outliers and patterns in the dataset  

### 2️⃣ Data Cleaning & Preprocessing
- Performed **train-test split** (80-20) with stratification  
- Replaced zeros in critical columns (`Glucose`, `BloodPressure`, `BMI`, `Insulin`) with **median values** from training set  
- Applied **log transformations** to skewed features (`Insulin`, `DiabetesPedigreeFunction`)  
- Scaled numerical features using `StandardScaler`  

### 3️⃣ Feature Engineering
- `BMI_Category`: Underweight / Normal / Overweight / Obese  
- `Glucose_Risk`: Normal / Pre-diabetic / Diabetic  
- `Age_Group`: Young / Middle-aged / Elderly  

### 4️⃣ Model Training & Evaluation
**Models Tested:**  
- Logistic Regression  
- Random Forest Classifier ✅ (Final model)  
- HistGradientBoostingClassifier  
- Support Vector Classifier (SVC)  
- XGBoost Classifier  

**Evaluation Metrics:**  
- **Recall** (primary metric)  
- F1-Score  
- ROC-AUC  
- Confusion Matrix & Classification Report  
- Cross-validation and threshold tuning for robust performance  

---

## 🏆 Chosen Model: Random Forest Classifier
Reasons for selection:  
- High recall and consistent F1-score across training and test sets  
- Minimal overfitting, strong generalization  
- Robust to feature scaling and missing values  


## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/Ronak9905/Diabetes-prediction.git
cd diabetes_prediction

# Install dependencies
pip install -r requirements.txt