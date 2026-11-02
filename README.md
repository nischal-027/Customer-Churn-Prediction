# Customer Churn Prediction using Machine Learning and ANN

Predicting whether a bank customer will leave (churn) using two approaches: a **Random Forest** classifier and an **Artificial Neural Network (ANN)** built with TensorFlow/Keras. Both models are trained on the same dataset and compared on the same evaluation metrics.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Key Observations](#key-observations)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)

---

## Project Overview

Customer churn is a major concern for banks, since retaining a customer is usually cheaper than acquiring a new one. This project builds binary classifiers that predict whether a customer will exit the bank (`Exited = 1`) or stay (`Exited = 0`), based on demographic and account information.

Two notebooks are included:

| Notebook | Approach |
|---|---|
| `Customer_Churn_Prediction_ML.ipynb` | Random Forest (scikit-learn) with exploratory analysis |
| `Customer_Churn_Prediction_ANN.ipynb` | Feed-forward neural network (TensorFlow / Keras) |

---

## Dataset

**File:** `Churn_Modelling.csv`: 10,000 customer records, 14 columns, no missing values.

| Column | Description |
|---|---|
| `RowNumber`, `CustomerId`, `Surname` | Identifiers (dropped before training) |
| `CreditScore` | Customer credit score |
| `Geography` | Country (France, Germany, Spain) |
| `Gender` | Male / Female |
| `Age` | Customer age |
| `Tenure` | Years as a bank customer |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Has a credit card (1/0) |
| `IsActiveMember` | Active member (1/0) |
| `EstimatedSalary` | Estimated salary |
| `Exited` | **Target**: 1 if the customer churned, 0 otherwise |

**Class distribution:** 7,963 stayed (79.6%) vs. 2,037 churned (20.4%), so the dataset is imbalanced.

> The dataset is not included in this repository. Place `Churn_Modelling.csv` in the project root (it is the widely used "Churn Modelling" dataset available on Kaggle) and update the file path in the notebooks.

---

## Repository Structure

```
.
├── Customer_Churn_Prediction_ML.ipynb    # Random Forest model
├── Customer_Churn_Prediction_ANN.ipynb   # Neural network model
├── Churn_Modelling.csv                   # Dataset (add manually)
└── README.md
```

---

## Methodology

### 1. Machine Learning Model (Random Forest)

1. **Exploration:** checked for null values, summary statistics, and the churn class distribution (count plot).
2. **Encoding:** `Gender` label-encoded; `Geography` one-hot encoded (`drop_first=True`).
3. **Feature selection:** dropped `RowNumber`, `CustomerId`, `Surname`, and the target, leaving 11 features.
4. **Split:** 80% train / 20% test, with `stratify=y` to preserve the churn ratio.
5. **Scaling:** `StandardScaler` fit on the training set only.
6. **Model:** `RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0)`.
7. **Evaluation:** accuracy, confusion matrix, and classification report.

### 2. Artificial Neural Network (ANN)

1. **Preprocessing:** same feature set; `Gender` label-encoded, `Geography` one-hot encoded via `ColumnTransformer`, features standardized.
2. **Split:** 80% train / 20% test (`random_state=0`).
3. **Architecture:**

   | Layer | Units | Activation |
   |---|---|---|
   | Hidden layer 1 | 6 | ReLU |
   | Hidden layer 2 | 6 | ReLU |
   | Output layer | 1 | Sigmoid |

4. **Training:** Adam optimizer, binary cross-entropy loss, batch size 32, 150 epochs.
5. **Prediction:** threshold of 0.5 on the sigmoid output; the notebook also demonstrates predicting for a single customer.
6. **Evaluation:** confusion matrix and classification report.

---

## Results

Both models were evaluated on a held-out test set of 2,000 customers.

| Metric | Random Forest | ANN |
|---|---|---|
| Test accuracy | 86.1% | 86.0% |
| Train accuracy | 90.1% | 86.1% (final epoch) |
| Precision (churn) | 0.80 | 0.73 |
| Recall (churn) | 0.43 | 0.51 |
| F1-score (churn) | 0.55 | 0.60 |
| Macro avg F1 | 0.74 | 0.76 |

---

## Key Observations

- **Similar overall accuracy.** Both models reach roughly 86% on the test set.
- **Accuracy hides the minority class.** Because only about 20% of customers churn, a model can score high accuracy while missing many churners. Both models have low recall on the churn class (43% for Random Forest, 51% for the ANN).
- **ANN catches more churners.** It has higher recall and F1 for the churn class, while Random Forest is more precise (fewer false alarms).
- **Random Forest overfits slightly.** Its training accuracy (90.1%) is noticeably above its test accuracy (86.1%).
- **Which is better depends on the goal.** If the priority is identifying as many at-risk customers as possible, the ANN's higher recall is preferable. If retention offers are expensive and false alarms matter, Random Forest's higher precision may be preferable.

> Note: the two notebooks use slightly different train/test splits (the ML notebook is stratified, the ANN notebook is not), so the comparison is approximate.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

---

## Usage

1. Download `Churn_Modelling.csv` and place it in the project folder.
2. In each notebook, update the dataset path, e.g.:
   ```python
   dataset = pd.read_csv('Churn_Modelling.csv')
   ```
   (The notebooks currently use the Google Colab path `/content/Churn_Modelling.csv`.)
3. Launch Jupyter and run the notebooks:
   ```bash
   jupyter notebook
   ```
   Or open them directly in Google Colab and upload the CSV.

---

## Future Improvements

- Address class imbalance (SMOTE, class weights, or threshold tuning) to improve churn recall.
- Hyperparameter tuning with `GridSearchCV` / `RandomizedSearchCV` and tuning the ANN's depth, width, and dropout.
- Add early stopping and validation curves to the ANN.
- Compare more models (Logistic Regression, XGBoost, SVM).
- Report ROC-AUC and precision-recall curves, and plot feature importances.
- Use the same stratified split for both models for a fully fair comparison.

---

## Tech Stack

- **Language:** Python
- **Data:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **ML:** scikit-learn (Random Forest, preprocessing, metrics)
- **Deep Learning:** TensorFlow / Keras

---

## Author

**Nischal Acharya**
