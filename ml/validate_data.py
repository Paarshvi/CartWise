import pandas as pd


FILE = "data/processed/cartwise_normalized.csv"

df = pd.read_csv(FILE)

print("===== DATA QUALITY REPORT =====")

print("\nDataset shape:")
print(df.shape)


# 1. Missing values
print("\nMissing values:")
print(df.isnull().sum())


# 2. Duplicate product IDs
print("\nDuplicate product IDs:")
print(df["product_id"].duplicated().sum())


# 3. Invalid prices
print("\nInvalid MRP values:")
print((df["mrp"] <= 0).sum())


# 4. Missing product names
print("\nMissing product names:")
print(df["normalized_name"].eq("").sum())


# 5. Quantity parsing
print("\nProducts with unparsed quantities:")
print(df["quantity_value"].isna().sum())


# 6. Platform distribution
print("\nProducts by platform:")
print(df["platform"].value_counts())


# 7. Category distribution
print("\nTop categories:")
print(df["category"].value_counts().head(10))


print("\n===== VALIDATION COMPLETE =====")