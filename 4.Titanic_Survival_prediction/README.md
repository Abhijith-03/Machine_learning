# 🚢 Titanic Survival Prediction — ML vs Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

*A complete binary classification study comparing classical machine learning and deep learning on the iconic Titanic dataset.*

</div>

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset Description](#-dataset-description)
3. [Project Structure](#-project-structure)
4. [End-to-End Pipeline](#-end-to-end-pipeline)
5. [EDA Key Findings](#-eda-key-findings)
6. [Preprocessing Steps](#-preprocessing-steps)
7. [Model Architectures](#-model-architectures)
8. [Results & Comparison](#-results--comparison)
9. [Key Insights & Conclusions](#-key-insights--conclusions)
10. [Learning Outcomes](#-learning-outcomes)
11. [How to Run](#-how-to-run)
12. [Requirements](#-requirements)
13. [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

This project tackles the classic **Titanic survival prediction** problem as a **binary classification** task — predicting whether a passenger survived (`1`) or did not survive (`0`) the Titanic disaster.

The core objective is **not just to predict**, but to **compare the performance of classical machine learning models against a custom-built deep learning model (MLP in PyTorch)** on a small, real-world tabular dataset.

| Property       | Detail                                  |
|----------------|-----------------------------------------|
| Task           | Binary Classification                   |
| Target         | `Survived` (0 = No, 1 = Yes)            |
| Dataset Size   | 891 rows × 12 columns                   |
| Train/Test     | 80% / 20% (stratified split)            |
| Baseline       | 61.60% (majority class)                 |

---

## 📊 Dataset Description

The dataset is the well-known [Titanic dataset](https://www.kaggle.com/c/titanic) containing passenger information.

| Feature       | Type        | Description                                      |
|---------------|-------------|--------------------------------------------------|
| `PassengerId` | Integer     | Unique passenger identifier                      |
| `Survived`    | Integer     | Target — 0 = No, 1 = Yes                         |
| `Pclass`      | Integer     | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)        |
| `Name`        | String      | Passenger's full name                            |
| `Sex`         | Categorical | Gender of the passenger                          |
| `Age`         | Float       | Age in years (177 missing values)                |
| `SibSp`       | Integer     | # of siblings/spouses aboard                    |
| `Parch`       | Integer     | # of parents/children aboard                    |
| `Ticket`      | String      | Ticket number                                    |
| `Fare`        | Float       | Passenger fare                                   |
| `Cabin`       | String      | Cabin number (687 missing — ~77%)                |
| `Embarked`    | Categorical | Port of embarkation (C = Cherbourg, Q, S)        |

**Missing Values Summary:**

| Column     | Missing Count | Missing % |
|------------|---------------|-----------|
| `Age`      | 177           | 19.87%    |
| `Cabin`    | 687           | 77.10%    |
| `Embarked` | 2             | 0.22%     |

---

## 📁 Project Structure

```
4.Titanic_Survival_prediction/
│
├── titanic.csv                  # Raw dataset
├── eda.ipynb                    # Exploratory Data Analysis notebook
├── preprocessing.ipynb          # Feature engineering & preprocessing
├── ml_models.ipynb              # Logistic Regression & Random Forest
├── deep_learning_mlp.ipynb      # PyTorch MLP model
└── README.md                    # Project documentation
```

---

## 🔄 End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW DATA (titanic.csv)                   │
│                       891 rows × 12 columns                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXPLORATORY DATA ANALYSIS                    │
│  • Missing value analysis    • Distribution plots               │
│  • Survival rate by group    • Correlation heatmap              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING                            │
│  • Drop: Name, Ticket, Cabin, PassengerId                       │
│  • Fill: Age (median), Embarked (mode)                          │
│  • Engineer: FamilySize = SibSp + Parch + 1                     │
│              IsAlone = 1 if FamilySize == 1 else 0              │
│  • ColumnTransformer:                                           │
│      – StandardScaler  → [Age, Fare, FamilySize]                │
│      – OneHotEncoder   → [Sex, Embarked, Pclass]                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
     ┌──────────────────────┐  ┌─────────────────────────┐
     │   CLASSICAL ML       │  │   DEEP LEARNING (MLP)   │
     │  Logistic Regression │  │   PyTorch — 3009 params  │
     │  Random Forest       │  │   BCELoss + Adam         │
     └──────────┬───────────┘  └──────────┬──────────────┘
                │                         │
                └──────────┬──────────────┘
                           │
                           ▼
           ┌───────────────────────────────┐
           │     EVALUATION & COMPARISON   │
           │  Accuracy, F1, AUC, Precision │
           │  Recall, Confusion Matrix     │
           └───────────────────────────────┘
```

---

## 🔍 EDA Key Findings

- **Survival Rate:** Only **38.38%** of passengers survived (549 did not, 342 did)
- **Sex:** Female passengers had a dramatically higher survival rate — *"women and children first"* confirmed in data
- **Pclass:** 1st class passengers survived at much higher rates than 3rd class
- **Age:** Children had relatively higher survival rates; middle-aged males had the lowest
- **Fare:** Higher fare strongly correlated with survival (linked to Pclass)
- **Embarked:** Passengers from Cherbourg (C) had the highest survival rate
- **Family Size:** Passengers traveling alone or in very large groups had lower survival rates

---

## ⚙️ Preprocessing Steps

1. **Drop irrelevant columns** — `Name`, `Ticket`, `Cabin`, `PassengerId`
2. **Impute missing values**
   - `Age` → filled with **median** age
   - `Embarked` → filled with **mode** (most frequent port)
3. **Feature Engineering**
   - `FamilySize = SibSp + Parch + 1` — total family members on board
   - `IsAlone = 1 if FamilySize == 1 else 0` — binary flag for solo travelers
4. **ColumnTransformer**
   - `StandardScaler` → `Age`, `Fare`, `FamilySize`
   - `OneHotEncoder` → `Sex`, `Embarked`, `Pclass`
5. **Stratified Train/Test Split** — 80/20 split preserving class balance

**Final feature count after encoding:** 10 input features

---

## 🧠 Model Architectures

### Logistic Regression
- Solver: `lbfgs`, Max iterations: 1000
- Regularization: L2 (default C=1.0)

### Random Forest
- Ensemble of decision trees
- Handles non-linearity and feature interactions natively
- Built-in feature importance

### MLP — PyTorch Architecture

```
Input Layer
    │
    │  (10 features)
    ▼
┌─────────────────────────┐
│    Linear(10 → 64)      │
│    BatchNorm1d(64)       │
│    ReLU                  │
│    Dropout(p=0.3)        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    Linear(64 → 32)      │
│    BatchNorm1d(32)       │
│    ReLU                  │
│    Dropout(p=0.3)        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    Linear(32 → 1)       │
│    Sigmoid              │
└────────────┬────────────┘
             │
             ▼
         Output
    (survival probability)
```

| Hyperparameter  | Value              |
|-----------------|--------------------|
| Loss Function   | BCELoss            |
| Optimizer       | Adam               |
| Learning Rate   | 0.001              |
| Weight Decay    | 1e-4               |
| Epochs          | 100                |
| Batch Size      | 32                 |
| Total Params    | 3,009              |

---

## 📈 Results & Comparison

| Model               | Accuracy  | Precision | Recall  | F1      | AUC    |
|---------------------|-----------|-----------|---------|---------|--------|
| Baseline (Majority) | 61.60%    | —         | —       | —       | 0.5000 |
| Logistic Regression | 80.45%    | 78.33%    | 68.12%  | 72.87%  | **0.8509** |
| Random Forest       | 80.45%    | 76.56%    | 71.01%  | **73.68%**  | 0.8337 |
| MLP (PyTorch)       | **81.56%**| **86.00%**| 62.32%  | 72.27%  | 0.8436 |

### 🏆 Winner Analysis

| Metric          | Best Model          | Value   |
|-----------------|---------------------|---------|
| **Accuracy**    | MLP (PyTorch)       | 81.56%  |
| **Precision**   | MLP (PyTorch)       | 86.00%  |
| **Recall**      | Random Forest       | 71.01%  |
| **F1 Score**    | Random Forest       | 73.68%  |
| **AUC**         | Logistic Regression | 0.8509  |

> **Verdict:** No single model dominates across all metrics. If **precision** matters (minimizing false positives), choose **MLP**. If **recall or F1** matters (catching more survivors), choose **Random Forest**. For overall **discrimination ability (AUC)**, **Logistic Regression** wins — and remains the most interpretable.

---

## 💡 Key Insights & Conclusions

- **Sex** and **Fare** were the strongest predictors of survival
- The engineered feature **`FamilySize`** added meaningful signal — solo travelers and very large families had lower survival odds
- **MLP achieved the highest precision (86%)** but the lowest recall — it is conservative in predicting survival
- **Classical ML matched or exceeded deep learning** on this small tabular dataset, reinforcing that DL is not always the best tool for structured data
- All three models **significantly beat the 61.6% majority baseline**, validating the modeling effort
- **Logistic Regression** offers the best balance of performance and interpretability

---

## 📚 Learning Outcomes

- End-to-end ML pipeline: EDA → Preprocessing → Modeling → Evaluation
- Handling real-world missing data with domain-aware strategies
- Feature engineering to extract latent signal (`FamilySize`, `IsAlone`)
- Using `ColumnTransformer` for clean, pipeline-compatible preprocessing
- Building and training a custom **MLP in PyTorch** with `BatchNorm` and `Dropout`
- Comparing classical ML and deep learning on tabular data
- Understanding **Precision vs. Recall trade-offs** in binary classification

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Abhijith-03/Machine-learning.git
cd "Machine-learning/4.Titanic_Survival_prediction"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebooks in order

```bash
# Step 1 — EDA
jupyter notebook eda.ipynb

# Step 2 — Preprocessing
jupyter notebook preprocessing.ipynb

# Step 3 — Classical ML models
jupyter notebook ml_models.ipynb

# Step 4 — PyTorch MLP
jupyter notebook deep_learning_mlp.ipynb
```

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
torch>=2.0.0
jupyter>=1.0.0
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch jupyter
```

---

## 🔮 Future Improvements

- [ ] **Hyperparameter tuning** — GridSearchCV for ML models, Optuna for MLP
- [ ] **Additional models** — XGBoost, LightGBM, SVM
- [ ] **Title extraction** — parse `Mr.`, `Mrs.`, `Miss.` from `Name` as a feature
- [ ] **Cabin deck** — extract deck letter from `Cabin` for passengers where available
- [ ] **Cross-validation** — k-fold CV for more robust performance estimates
- [ ] **SHAP values** — explainability analysis for the Random Forest and MLP
- [ ] **Ensemble / Stacking** — combine all three models for potentially higher performance
- [ ] **Kaggle submission** — evaluate on the official Titanic test set

---

<div align="center">

Made with ❤️ by **Abhijith**

*"Not all who wandered were lost — but with ML, we can predict who might have been."*

</div>
