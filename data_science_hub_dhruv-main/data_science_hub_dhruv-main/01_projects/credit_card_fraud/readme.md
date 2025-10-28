# Financial Fraud Detection Using Machine Learning

A comprehensive machine learning solution for detecting fraudulent financial transactions with **99.63% accuracy**, achieving a **51,044X improvement** over traditional rule-based systems.

---

## Project Overview

This project analyzes **6.36 million synthetic financial transactions** to build and compare multiple machine learning models for fraud detection. The solution addresses the critical challenge of highly imbalanced data (fraud rate: 0.13%) and delivers production-ready fraud detection capabilities.

### Key Achievements
- **99.63% Fraud Detection Rate** (Recall)
- **98.26% Precision** (minimal false positives)
- **98.94% F1-Score** (best overall performance)
- **51,044X improvement** over existing flagging system
- Trained and compared **8 different ML models**
- Handled severe class imbalance (**1:774 fraud-to-legitimate ratio**)

---

## Problem Statement

Financial institutions lose billions annually to fraud. Traditional rule-based systems are:
- Ineffective (catching only **0.19%** of fraud cases)
- Rigid (cannot adapt to new fraud patterns)
- Resource-intensive (manual review of false positives)

**This project demonstrates how modern ML can revolutionize fraud detection.**

---

## Dataset

**Source:** PaySim Synthetic Financial Dataset (Kaggle)  
**Size:** 6,362,620 transactions  
**Features:** 11 columns  
**Target:** Binary classification (Fraud vs Legitimate)

### Dataset Characteristics:
- **Severe imbalance:** 99.87% legitimate, 0.13% fraud (8,213 fraud cases)
- **Transaction types:** PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT
- **Time span:** 744 time steps (~30 days, 1 step = 1 hour)
- **No missing values** (clean dataset)

### Key Discovery:
**Fraud only occurs in 2 transaction types:**
- TRANSFER: 0.77% fraud rate
- CASH_OUT: 0.18% fraud rate
- Zero fraud in: PAYMENT, CASH_IN, DEBIT

---

## Technologies Used

### Core Stack:
- **Python 3.x**
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **Imbalanced-learn (SMOTE)** - Handling class imbalance
- **Matplotlib & Seaborn** - Data visualization

### ML Algorithms Tested:
1. Gradient Boosting
2. Random Forest
3. XGBoost
4. Decision Tree
5. AdaBoost
6. Logistic Regression
7. K-Nearest Neighbors
8. Naive Bayes

---

## Methodology

### 1. **Exploratory Data Analysis**
- Identified fraud patterns by transaction type
- Discovered 65% of frauds target zero-balance accounts
- Found fraudulent amounts average 8X larger ($1.47M vs $178K)
- Analyzed 2.6M balance anomalies as fraud indicators

### 2. **Feature Engineering**
Created 15 fraud-indicating features:
- **Balance anomalies** (origin/destination errors)
- **Zero-balance flags** (before/after transaction)
- **Amount transformations** (log scaling, ratios)
- **Transaction type encoding**
- **Temporal features** (hour of day patterns)

**Most Important Features:**
1. `error_orig` (84.98% importance) - Balance calculation errors
2. `newbalanceOrig` (12.59%) - Post-transaction balance
3. `amount` (1.11%) - Transaction amount

### 3. **Handling Class Imbalance**
- **SMOTE (Synthetic Minority Over-sampling)** to balance training data
- Increased fraud samples from 0.3% to 50% in the training set
- Applied `class_weight='balanced'` to tree-based models
- Maintained the original test set for realistic evaluation

### 4. **Model Training & Evaluation**
- 80/20 train-test split (stratified)
- Trained 8 models for comprehensive comparison
- Evaluated using fraud-specific metrics (Precision, Recall, F1, ROC-AUC)
- Selected Gradient Boosting as the champion model

---

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| **Gradient Boosting** | **99.99%** | **98.26%** | **99.63%** | **98.94%** | **99.84%** | 69 min |
| Random Forest | 99.99% | 97.97% | 99.63% | 98.79% | 99.90% | 12 min |
| XGBoost | 99.99% | 97.67% | 99.63% | 98.64% | 99.85% | 26 sec |
| Decision Tree | 99.99% | 97.09% | 99.63% | 98.35% | 99.82% | 48 sec |
| AdaBoost | 99.98% | 95.17% | 99.63% | 97.35% | 99.84% | 10 min |
| KNN | 99.22% | 26.34% | 90.75% | 40.83% | 95.86% | 41 sec |
| Naive Bayes | 97.05% | 4.91% | 48.81% | 8.93% | 88.00% | 2 sec |
| Logistic Regression | 94.14% | 4.40% | 90.57% | 8.39% | 98.04% | 4 min |

### Best Model: Gradient Boosting
- **Catches 1,637 out of 1,643 frauds** (99.63% recall)
- **98.26% precision** (only 29 false alarms out of 1,666 fraud predictions)
- **6 frauds missed** (0.37% false negative rate)
- **Near-perfect ROC-AUC:** 99.84%

### Business Impact

| Metric | Old System | New Model | Improvement |
|--------|-----------|-----------|-------------|
| Detection Rate | 0.19% | 99.63% | **51,044X** |
| Frauds Caught | 16 | 1,637 | +1,621 |
| Frauds Missed | 8,197 | 6 | -8,191 |
| F1-Score | 0.39% | 98.94% | 253X |

**Financial Impact:**
- With an average fraud loss of $1.47M per case
- Preventing 1,621 additional frauds = **$2.38 BILLION saved** (on test set alone)

---

## Key Insights

### 1. **Fraud Patterns Discovered**
-  **Amount:** Fraudulent transactions average **$1.47M** vs **$178K** legitimate
-  **Zero Balances:** 65% of fraud target accounts with $0 initial balance
-  **Transaction Type:** TRANSFER is **4.2X riskier** than CASH_OUT
-  **Balance Errors:** 84.98% of fraud detection relies on balance calculation anomalies

### 2. **Model Selection Insights**
- **Tree-based models dominate** (top 5 are all ensemble methods)
- **XGBoost offers best efficiency:** 98.64% F1-Score in just 26 seconds
- **Gradient Boosting is most accurate** but takes 69 minutes to train
- **Simple models fail:** Logistic Regression, Naive Bayes, KNN perform poorly

### 3. **Feature Importance**
Balance errors (`error_orig`) account for **85%** of fraud detection signal, confirming that fraudsters manipulate account balances in detectable ways.

---

## Project Structure
fraud-detection/
├── data/
│   └── PS_20174392719_1491204439457_log.csv
├── notebooks/
│   └── fraud_detection_analysis.ipynb
├── visualizations/
│   ├── fraud_viz_01_model_comparison.png
│   ├── fraud_viz_02_precision_recall_tradeoff.png
│   ├── fraud_viz_03_confusion_matrix.png
│   ├── fraud_viz_04_feature_importance.png
│   ├── fraud_viz_05_roc_curves.png
│   ├── fraud_viz_06_amount_distribution.png
│   ├── fraud_viz_07_transaction_type_analysis.png
│   ├── fraud_viz_08_time_vs_performance.png
│   └── fraud_viz_09_business_impact.png
├── models/
│   └── gradient_boosting_best_model.pkl
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
```

### 3. Download the dataset
- Download from [Kaggle PaySim Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
- Place in `data/` folder

### 4. Run the analysis
```bash
jupyter notebook notebooks/fraud_detection_analysis.ipynb
```

---

## Visualizations

### Model Performance Comparison
![Model Comparison](visualizations/fraud_viz_01_model_comparison.png)
*Gradient Boosting achieves the highest F1-Score at 98.94%*

### Business Impact
![Business Impact](visualizations/fraud_viz_09_business_impact.png)
*51,044X improvement over the old system - from 0.19% to 99.63% detection rate*

### Feature Importance
![Feature Importance](visualizations/fraud_viz_04_feature_importance.png)
*Balance errors (error_orig) are the strongest fraud indicator at 84.98%*

### ROC Curves
![ROC Curves](visualizations/fraud_viz_05_roc_curves.png)
*All top models achieve near-perfect ROC-AUC scores above 99.8%*

---

## Key Takeaways

1. **Class imbalance is solvable:** SMOTE + proper evaluation metrics enable accurate fraud detection despite 1:774 imbalance

2. **Feature engineering matters:** Balance anomalies proved more valuable than amount or transaction type alone

3. **Tree-based models excel at fraud:** Ensemble methods capture complex non-linear fraud patterns better than traditional algorithms

4. **Business impact is massive:** 99.63% detection rate translates to billions in prevented losses

5. **Efficiency trade-offs exist:** XGBoost offers 98.64% accuracy in 26 seconds vs Gradient Boosting's 98.94% in 69 minutes

---

##  Learning Outcomes

Through this project, I demonstrated expertise in:
- **Handling severe class imbalance** (1:774 ratio)
- **Feature engineering** for fraud detection
- **Model comparison** across 8 algorithms
- **Hyperparameter tuning** for optimal performance
- **Business impact analysis** and cost-benefit assessment
- **Data visualization** for stakeholder communication
- **Production-ready ML** pipeline development

---

## Future Enhancements

1. **Deep Learning:** Experiment with Neural Networks (LSTM for sequential patterns)
2. **Real-time Deployment:** Build API for live fraud scoring
3. **Explainable AI:** Implement SHAP values for model interpretability
4. **Cost-sensitive Learning:** Optimize for business costs (false positive vs false negative)
5. **Anomaly Detection:** Explore unsupervised methods (Isolation Forest, Autoencoders)
6. **Time-series Features:** Capture customer behavior patterns over time

---

## Contact

**Dhruv Sharma**  
-  Email: [dhruvsreadingnook@gmail.com]
-  LinkedIn: [linkedin.com/in/dhruv-sharmaokstate]
-  GitHub: [github.com/madwolfslayer]
-  Medium: [medium.com/@dhruvsharma14]

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Acknowledgments

- **Dataset:** PaySim synthetic financial dataset from Kaggle
- **Inspiration:** Real-world fraud detection challenges in financial services
- **Tools:** Scikit-learn, XGBoost, and the open-source ML community

---

** If you found this project helpful, please give it a star!**

** Questions or feedback? Open an issue or reach out!**
