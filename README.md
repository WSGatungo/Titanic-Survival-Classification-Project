# Titanic: Machine Learning from Disaster 🚢

An end-to-end data science project predicting passenger survival on the Titanic using classification algorithms. This project focuses on rigorous exploratory data analysis (EDA), custom feature engineering, and a performance comparison between ensemble models and gradient boosting.

## 📊 Project Overview
The goal of this project is to predict whether a passenger survived the Titanic shipwreck (`1` for survived, `0` for deceased) based on features like age, socio-economic status, gender, and family structure.

### Key Insights from Data Analysis
* **Gender:** `Sex` was the single strongest predictor. Women had a significantly higher survival rate, aligning with historical "women and children first" protocols.
* **Socio-economic Status:** A clear gradient was visible across passenger classes. 1st Class passengers had a >60% survival probability, whereas 3rd Class passengers dropped below 25%.
* **Age:** Children (ages 0-5) experienced a distinct spike in survival rates compared to any other demographic.

---

## 🛠️ Feature Engineering & Preprocessing
Rather than feeding raw data directly into the models, several domain-specific features were engineered to capture hidden patterns:
* **Title Extraction:** Extracted social titles (*Mr., Mrs., Miss, Master*) from the `Name` column to impute missing `Age` values more precisely by using group medians.
* **Cabin Indicator:** Transformed the sparse `Cabin` column into a binary feature (`Has_Cabin`) to capture the correlation between recorded data and proximity to lifeboats.
* **Family Size Matrix:** Combined `SibSp` (siblings/spouses) and `Parch` (parents/children) into a unified `FamilySize` metric. Small families (2-4 people) showed higher survival rates than individuals traveling alone or large families.
* **Categorical Encoding:** Applied One-Hot Encoding via `pd.get_dummies(drop_first=True)` to safely convert text categorical variables into numerical binaries without introducing multicollinearity.

---

## 🤖 Model Comparison & Results
Three distinct approaches were evaluated on a fixed 20% validation split to determine the most reliable architecture for this data scale:

| Model | Validation Accuracy | Notes / Parameters |
| :--- | :--- | :--- |
| **1. Baseline Random Forest** | **82.68%** | Default parameters, `max_depth=5` |
| **2. XGBoost Classifier** | **82.12%** | `learning_rate=0.05`, `max_depth=4` |
| **3. GridSearch Tuned RF** | **81.56%** | Optimized via 5-Fold Cross-Validation |

### Best Hyperparameters Found (Random Forest)
```python
{
    'criterion': 'entropy', 
    'max_depth': 6, 
    'min_samples_split': 10, 
    'n_estimators': 50
}
