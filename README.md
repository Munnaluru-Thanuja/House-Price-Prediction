import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
!unzip -o house-prices-advanced-regression-techniques.zip
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")
train_df["TotalBathrooms"] = train_df["FullBath"] + train_df["HalfBath"]
test_df["TotalBathrooms"]  = test_df["FullBath"] + test_df["HalfBath"]
features = ["GrLivArea", "BedroomAbvGr", "TotalBathrooms"]
target = "SalePrice"
train_df[features] = train_df[features].fillna(train_df[features].mean())
test_df[features]  = test_df[features].fillna(test_df[features].mean())
train_df[target] = train_df[target].fillna(train_df[target].mean())
X = train_df[features]
y = train_df[target]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print("Model Performance")
print("------------------")
print("RMSE:", rmse)
print("R² Score:", r2)
test_predictions = model.predict(test_df[features])
output = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": test_predictions
})
output.to_csv("submission.csv", index=False)
print("\n SUCCESS — submission.csv created!")
