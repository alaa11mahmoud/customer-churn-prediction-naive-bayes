Customer Churn Prediction using Naive Bayes

This project uses the Naive Bayes classification algorithm to predict customer churn based on historical customer data. It aims to identify patterns in customer behavior that are likely to lead to churn, helping businesses take proactive actions.

📈 Workflow

1. **Data Cleaning**
   - Handled missing values, encoded categorical variables
   - Removed irrelevant features (like customer ID)

2. **Exploratory Data Analysis (EDA)**
   - Visualized churn rate by gender, tenure, contract type
   - Correlation analysis

3. **🧰 Feature Engineering**
   - Label encoding for binary variables
   - One-hot encoding for categorical variables

4. **Modeling with Naive Bayes**
   - Used `GaussianNB` from `scikit-learn`
   - Split data into training and test sets (80/20)
   - Trained model and made predictions

5. ** ✅ Evaluation**
   - Confusion Matrix
   - Accuracy, Precision, Recall, F1-Score
   - ROC Curve & AUC

6. **🖥️ Deployment Model**
   - streamlit


🛠 Tools & Technologies
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook
