import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Load Data
file_path = r"C:\Users\Admin\Documents\python project\Mutual_Funds_500.csv"
data = pd.read_csv(file_path)

# 2. CLEANING DATA (Crucial step to fix your error)
# Remove currency symbols and formatting from numeric and categorical columns
data['Min. Investment'] = data['Min. Investment'].str.replace('₹', '').str.replace(',', '').astype(float)
data['Exp. Return'] = data['Exp. Return'].str.replace('%', '').astype(float)

# NEW: Clean 'Min. Income' so it matches your function input ('3L+' instead of '₹3L+')
data['Min. Income'] = data['Min. Income'].str.replace('₹', '')

# Filter out rare funds to improve learning
counts = data['Bank / Company'].value_counts()
data = data[data['Bank / Company'].isin(counts[counts >= 3].index)].copy()

# 3. Encoding
encoders = {}
categorical_cols = ['Age Group', 'Duration', 'Risk', 'Investment Type', 'Sector/Industry', 'Min. Income']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

target_le = LabelEncoder()
data['Bank / Company'] = target_le.fit_transform(data['Bank / Company'])

# 4. Train Model
X = data.drop(columns=['Bank / Company'])
y = data['Bank / Company']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# 5. Recommendation Function
def recommend_fund(age_group, min_invest, duration, risk, exp_return, invest_type, sector, min_income):
    # Prepare input data
    # Note: We use the same cleanup here if needed, but since we are passing '3L+',
    # and the encoder now knows '3L+', it will work.
    try:
        input_row = pd.DataFrame([[
            encoders['Age Group'].transform([age_group])[0],
            min_invest,
            encoders['Duration'].transform([duration])[0],
            encoders['Risk'].transform([risk])[0],
            exp_return,
            encoders['Investment Type'].transform([invest_type])[0],
            encoders['Sector/Industry'].transform([sector])[0],
            encoders['Min. Income'].transform([min_income])[0]
        ]], columns=X.columns)

        probs = model.predict_proba(input_row)[0]
        top_3_idx = np.argsort(probs)[-3:][::-1]
        return target_le.inverse_transform(top_3_idx)
    except ValueError as e:
        return f"Error: One of the inputs was not recognized. {e}"


import joblib

# Save everything to files
joblib.dump(model, 'mutual_fund_model.pkl')
joblib.dump(encoders, 'feature_encoders.pkl')
joblib.dump(target_le, 'target_encoder.pkl')