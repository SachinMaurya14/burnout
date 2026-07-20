import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# 1. Load data
data_path = "student-lifestyle-and-stress-dataset.csv"
df = pd.read_csv(data_path)

# 2. Impute missing values (matching the notebook)
num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. Clean string columns
for col in cat_cols:
    df[col] = df[col].str.strip().str.title()

# 4. Drop Month
if 'Month' in df.columns:
    df.drop('Month', axis=1, inplace=True)

# 5. Split x and y
x = df.drop(['Stress_Level'], axis=1)
y = df['Stress_Level']

# 6. Map Student_Type
x['Student_Type'] = x['Student_Type'].map({
    'School': 0, 'College': 1, 'Working_Student': 2
})

# 7. Load scaler and scale features
scaler_path = "scaler.pkl"
scaler = joblib.load(scaler_path)
X_scaled = scaler.transform(x)

# 8. Train Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)

# 9. Save trained model
joblib.dump(model, "model.pkl")
print("Model trained and saved successfully as model.pkl")
