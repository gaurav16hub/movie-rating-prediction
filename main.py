import matplotlib.pyplot as plt
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("IMDb Movies India.csv", encoding="latin-1")
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Clean numerical columns
for col in ["Year", "Duration", "Votes", "Rating"]:
    if col in df.columns:
        if col == "Votes":
            df[col] = df[col].astype(str).str.replace(
                ",", "", regex=False
            )

        df[col] = pd.to_numeric(
            df[col].astype(str).str.extract(
                r"(\d+(?:\.\d+)?)"
            )[0],
            errors="coerce"
        )

# 3. Remove rows without a target rating
df = df.dropna(subset=["Rating"])
# Remove duplicate movies and invalid ratings

df = df[df["Rating"].between(0, 10)]

print("Cleaned dataset shape:", df.shape)
print("Missing ratings:", df["Rating"].isna().sum())

# 4. Select features
features = [
    "Year", "Duration", "Votes", "Genre",
    "Director", "Actor 1", "Actor 2", "Actor 3"
]

features = [col for col in features if col in df.columns]

X = df[features]
y = df["Rating"]

numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = [
    col for col in X.columns if col not in numeric_cols
]

# 5. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# 6. Build model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train
print("\nTraining model...")
model.fit(X_train, y_train)

# 9. Evaluate
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\n--- Model Evaluation ---")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2 Score: {r2:.3f}")

# 10. Compare actual and predicted ratings
results = pd.DataFrame({
    "Actual Rating": y_test.values,
    "Predicted Rating": predictions.round(2)
})

print("\nSample predictions:")
print(results.head(10).to_string(index=False))

results.to_csv("prediction_results.csv", index=False)
print("\nResults saved to prediction_results.csv")
# Compare actual and predicted ratings
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.5
)

plt.plot([0, 10], [0, 10], linestyle="--")

plt.xlabel("Actual Movie Rating")
plt.ylabel("Predicted Movie Rating")
plt.title("Actual vs Predicted Movie Ratings")

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()