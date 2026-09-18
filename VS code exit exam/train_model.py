import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


df = pd.read_csv("G3.csv")


drop_cols = [
    "enterprise_group_id",
    "display_name",
    "enterprise_group_name",
    "group_root_lei",
    "website",
    "employees_source",
    "revenue_source",
    "net_income_source",
    "market_cap_source",
    "employees_year",
    "revenue_year",
    "net_income_year",
    "market_cap_date",
    "revenue_currency",
    "net_income_currency",
    "market_cap_currency",
]
df_processed = df.drop(columns=drop_cols, errors="ignore")


num_cols = [
    "employees",
    "revenue",
    "net_income",
    "market_cap",
    "legal_units_count",
    "direct_subsidiaries_count",
    "max_hierarchy_depth",
]
for col in num_cols:
  if col in df_processed.columns:
    df_processed[col] = df_processed[col].fillna(df_processed[col].median())

cat_cols = [
    "company_type",
    "enterprise_group_entity_status",
    "enterprise_group_jurisdiction",
]
for col in cat_cols:
  if col in df_processed.columns:
    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

if "is_standalone_entity" in df_processed.columns:
  df_processed["is_standalone_entity"] = df_processed[
      "is_standalone_entity"
  ].astype(int)


df_encoded = pd.get_dummies(df_processed, columns=cat_cols, drop_first=True)


X = df_encoded.drop("revenue", axis=1)
y = df_encoded["revenue"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)


joblib.dump({"model": model, "feature_names": list(X.columns)}, "model.pkl")
print("Model saved to model.pkl")