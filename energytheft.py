# --------------------------
# Energy Theft Detection with Probability & Dashboard
# --------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# --------------------------
# Step 1: Create or Load Large Synthetic Dataset
# --------------------------
if not os.path.exists("energy_large_data.csv"):
    np.random.seed(42)
    num_customers = 10000
    months = 12

    data_list = []
    for cust_id in range(1, num_customers+1):
        for month in range(1, months+1):
            base = np.random.normal(300, 50)
            seasonal_factor = 20 * np.sin((month/12) * 2 * np.pi)
            consumption = base + seasonal_factor
            theft = np.random.choice([0,1], p=[0.95,0.05])
            if theft == 1:
                consumption *= np.random.uniform(0.2,0.7)
            data_list.append([cust_id, month, round(consumption,2), theft])

    df = pd.DataFrame(data_list, columns=['customer_id', 'month', 'consumption_kwh', 'is_theft'])
    df.to_csv("energy_large_data.csv", index=False)
else:
    df = pd.read_csv("energy_large_data.csv")

print("Dataset loaded. Sample:")
print(df.head())

# --------------------------
# Step 2: Preprocessing
# --------------------------
X = df[['month','consumption_kwh']]
y = df['is_theft']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --------------------------
# Step 3: Train Random Forest Model
# --------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --------------------------
# Step 4: Evaluate Model
# --------------------------
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

feat_importance = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(6,4))
sns.barplot(x=feat_importance, y=feat_importance.index)
plt.title("Feature Importance")
plt.show()

# --------------------------
# Step 5: Predict New Customer with Probability & Vacation Handling
# --------------------------
print("\nPredict new customer usage (probability-based):")
month = int(input("Month (1-12): "))
consumption = float(input("Consumption in kWh: "))
vacation = input("Was the customer on vacation this month? (yes/no): ").lower()

new_data = pd.DataFrame([[month, consumption]], columns=['month','consumption_kwh'])
prob_theft = model.predict_proba(new_data)[0][1]  # probability of theft

# Adjust probability if vacation
if vacation == 'yes':
    prob_theft *= 0.3  # reduce probability by 70% if vacation

# Threshold for alert
threshold = 0.7
if prob_theft >= threshold:
    print(f"⚠️ Alert: Possible energy theft detected! (Probability: {prob_theft:.2f})")
else:
    print(f"✅ Normal consumption. (Probability of theft: {prob_theft:.2f})")

# --------------------------
# Step 5b: Simple Console + Plot Dashboard for Risk Visualization
# --------------------------

# Predict probabilities for all customers in the dataset
df['pred_prob_theft'] = model.predict_proba(df[['month','consumption_kwh']])[:,1]

# Apply vacation adjustment (simulated here as random yes/no)
np.random.seed(42)
df['vacation'] = np.random.choice(['yes','no'], size=len(df))
df['adjusted_prob'] = df.apply(lambda row: row['pred_prob_theft']*0.3 if row['vacation']=='yes' else row['pred_prob_theft'], axis=1)

# Assign risk status based on probability thresholds
def risk_status(prob):
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    else:
        return "Normal"

df['risk_status'] = df['adjusted_prob'].apply(risk_status)

# Console Table for Dashboard (first 20 rows)
print("\n--- Risk Dashboard (Sample) ---")
print(df[['customer_id','month','consumption_kwh','adjusted_prob','risk_status']].head(20))

# Color-coded Bar Plot for Risk Distribution
plt.figure(figsize=(10,5))
status_counts = df['risk_status'].value_counts()
colors = ['green' if s=='Normal' else 'yellow' if s=='Medium Risk' else 'red' for s in status_counts.index]
plt.bar(status_counts.index, status_counts.values, color=colors)
plt.title("Customer Risk Distribution")
plt.xlabel("Risk Status")
plt.ylabel("Number of Records")
plt.show()

# Highlight top 20 high-risk customers
top_high_risk = df[df['risk_status']=='High Risk'].sort_values('adjusted_prob', ascending=False).head(20)
print("\n--- Top 20 High-Risk Customers ---")
print(top_high_risk[['customer_id','month','consumption_kwh','adjusted_prob']])

# --------------------------
# Step 6: Save Model
# --------------------------
joblib.dump(model, "energy_theft_large_model.pkl")
print("\nModel saved as 'energy_theft_large_model.pkl'")
