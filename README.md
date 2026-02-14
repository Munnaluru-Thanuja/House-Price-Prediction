# House-Price-Prediction
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# Step 1 — Load Data
# ---------------------------

# Unzip the uploaded archive to access the CSV files
# This command extracts all files from the zip into the current directory
!unzip -o house-prices-advanced-regression-techniques.zip

# Load data
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

# ---------------------------
# Step 2 — Feature Engineering
# ---------------------------

# Create TotalBathrooms feature
train_df["TotalBathrooms"] = train_df["FullBath"] + train_df["HalfBath"]
test_df["TotalBathrooms"]  = test_df["FullBath"] + test_df["HalfBath"]

features = ["GrLivArea", "BedroomAbvGr", "TotalBathrooms"]
target = "SalePrice"

# ---------------------------
# Step 3 — Handle Missing Values
# (Safer than dropna)
# ---------------------------

train_df[features] = train_df[features].fillna(train_df[features].mean())
test_df[features]  = test_df[features].fillna(test_df[features].mean())

# Target column missing values (rare but safe)
train_df[target] = train_df[target].fillna(train_df[target].mean())

X = train_df[features]
y = train_df[target]

# ---------------------------
# Step 4 — Train/Test Split
# ---------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Step 5 — Train Model
# ---------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------
# Step 6 — Evaluate
# ---------------------------

y_pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print("Model Performance")
print("------------------")
print("RMSE:", rmse)
print("R² Score:", r2)

# ---------------------------
# Step 7 — Predict Test Data
# ---------------------------

test_predictions = model.predict(test_df[features])

# ---------------------------
# Step 8 — Save Submission
# ---------------------------

output = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": test_predictions
})

output.to_csv("submission.csv", index=False)

print("\n✅ SUCCESS — submission.csv created!")
