# CREDIT CARD FRAUD DETECTION — Original project code + added project requirements
# Added sections are marked with:  # ADDED
# Your existing code sections are retained; see the explanation for where additions fit.

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

# ADDED: cross-validation and model-comparison tools
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score

# ADDED: outlier comparison helper
from scipy.stats import zscore

df=pd.read_csv("filtered_1000_records.csv")
print(df.head)
print(df.info)
print(df.describe)

# ADDED: DATA QUALITY CHECKS (reports missing values and duplicates)
print("\n===== ADDED: DATA QUALITY CHECK =====")
print("Missing values per column:\n", df.isnull().sum())
print("Total duplicate rows:", df.duplicated().sum())

# ADDED: missing-value handling report
# This creates a separate cleaned copy for the report; your original df is not overwritten.
df_quality = df.copy()
missing_before = int(df_quality.isnull().sum().sum())
df_quality = df_quality.drop_duplicates()
duplicates_removed = len(df) - len(df_quality)
print("Missing cells before handling:", missing_before)
print("Duplicate rows removed in report copy:", duplicates_removed)

# ADDED: EDA CHARTS — histogram, box plot, pie chart, bar chart, correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\n===== ADDED: EXPLORATORY DATA ANALYSIS =====")

# Histogram: transaction amount distribution
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="amt", bins=30, kde=True)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Box plot: amount outliers
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="amt")
plt.title("Transaction Amount Box Plot")
plt.tight_layout()
plt.show()

# Pie chart: genuine vs fraud class distribution
plt.figure(figsize=(6, 6))
df["is_fraud"].value_counts().sort_index().plot(
    kind="pie",
    autopct="%1.1f%%",
    labels=["Genuine (0)", "Fraud (1)"],
    startangle=90
)
plt.title("Genuine vs Fraud Transactions")
plt.ylabel("")
plt.tight_layout()
plt.show()

# Bar chart: fraud count by transaction category
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="category", hue="is_fraud")
plt.title("Fraud vs Genuine by Category")
plt.xlabel("Transaction Category")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Correlation matrix: numeric columns only
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# ADDED: OUTLIER REPORT (IQR method) — does not delete/change rows in the model dataset
print("\n===== ADDED: OUTLIER CHECK =====")
outlier_report = {}
for col in df.select_dtypes(include=np.number).columns:
    if col == "is_fraud":
        continue
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((df[col] < lower) | (df[col] > upper)).sum())
    outlier_report[col] = count
print("Outlier count by numeric column (IQR method):")
print(pd.Series(outlier_report, name="outlier_count"))

# ADDED: AUTOMATED EDA HTML REPORT (Sweetviz)
# Install once in VS Code terminal if needed: python -m pip install sweetviz
try:
    import sweetviz as sv
    report = sv.analyze(df)
    report.show_html("EDA_Report.html", open_browser=False)
    print("Automated EDA report saved as EDA_Report.html")
except ImportError:
    print("Sweetviz is not installed. To create the HTML report, run:")
    print("python -m pip install sweetviz")
except Exception as e:
    print("Automated EDA report could not be created:", e)

X=df.drop('is_fraud',axis=1)
Y=df['is_fraud']

print(X.shape)
print(Y.value_counts)

X_train,X_test,Y_train,Y_test =train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print("Training:",X_train.shape)
print("Testing",X_test.shape)


categorical_col=["merchant","category","city",'state',] #qualitative data

numerical_col=['cc_num', 'amt', 'zip', 'lat', 'long',
    'merch_lat', 'merch_long']                          #quanitative data

print("Categorical data:",categorical_col)
print("Numerical data",numerical_col)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_col),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_col)
    ]
)

model=Pipeline([
    ('preprocessor',preprocessor),
    ('classifire',LogisticRegression(max_iter=1000))
]
)

model.fit(X_train,Y_train)

print("Model training complete!")

y_pred=model.predict(X_test)

print("prediction Complete!")
print(y_pred)

accuracy=accuracy_score(Y_test,y_pred)

print("Accuracy:",accuracy)
print("\nClassification report")
print(classification_report(Y_test,y_pred))

print("\nConfusion metrix:")
print(confusion_matrix(Y_test,y_pred))

# ADDED: explicit precision, recall and F1 score summary
print("\n===== ADDED: CLASSIFICATION METRICS =====")
print("Precision:", precision_score(Y_test, y_pred, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred, zero_division=0))

# Your original confusion-matrix chart
cm=confusion_matrix(Y_test,y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=['Genuine(0)','Fraud(1)'],
    yticklabels=['Genuine(0)','Fraud(1)']
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Credit Card Fraud Detection - Confusion Matrix")
plt.show()

# ADDED: CROSS-VALIDATION — evaluates the existing pipeline across 5 folds
print("\n===== ADDED: 5-FOLD CROSS-VALIDATION =====")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, Y_train, cv=cv, scoring="accuracy")
print("Fold accuracies:", cv_scores)
print("Mean cross-validation accuracy:", cv_scores.mean())

# ADDED: REGULARIZATION COMPARISON — compares Logistic Regression C values.
# Smaller C means stronger regularization; larger C means weaker regularization.
print("\n===== ADDED: REGULARIZATION COMPARISON =====")
for c_value in [0.01, 0.1, 1.0, 10.0]:
    comparison_model = Pipeline([
        ('preprocessor',preprocessor),
        ('classifire',LogisticRegression(max_iter=1000, C=c_value))
    ])
    comparison_model.fit(X_train, Y_train)
    comparison_pred = comparison_model.predict(X_test)
    print(
        f"C={c_value:<5} | "
        f"Accuracy={accuracy_score(Y_test, comparison_pred):.3f} | "
        f"Precision={precision_score(Y_test, comparison_pred, zero_division=0):.3f} | "
        f"Recall={recall_score(Y_test, comparison_pred, zero_division=0):.3f} | "
        f"F1={f1_score(Y_test, comparison_pred, zero_division=0):.3f}"
    )

import joblib

joblib.dump(model, "fraud_detection_model.pkl")

print("Model saved successfully!")

print("\n--- Fraud Detection ---")
print("Enter details separated by commas:")
print("cc_num, merchant, category, amt, city, state, zip, lat, long, merch_lat, merch_long")


while True:
    print("====--|CREDIT CARD FRAUD DETECTION|--=====")
    print("1. Cheack New Transaction")
    print("2.Exit") 


    choice= input("Enter a choice:")
    if choice == "1":
        print("\nPaste your transactions (one transaction per line):")

        transactions = []
        while True:
              line = input()
              if line == "":
                    break
              transactions.append(line.split(","))
              for data in transactions:
                  new_data = pd.DataFrame([{
                      'cc_num': data[0].strip(),
                      'merchant': data[1].strip(),
                      'category': data[2].strip(),
                      'amt': float(data[3].strip()),
                      'city': data[4].strip(),
                      'state': data[5].strip(),
                      'zip': int(data[6].strip()),
                      'lat': float(data[7].strip()),
                      'long': float(data[8].strip()),
                      'merch_lat': float(data[9].strip()),
                      'merch_long': float(data[10].strip()
                                          )}])
                  prediction = model.predict(new_data)[0]
                  probability=model.predict_proba(new_data)[0][1]
                  if prediction == 1:
                      print("\nPrediction: FRAUD 🚨")
                  else:
                      print("\nPrediction: GENUINE ✅")

                  print(f"fraud Probability:{probability *100:.2f}%")
    elif choice=="2":
        print("Program closed:")
        break

    else:
        print("invelid choice!")
