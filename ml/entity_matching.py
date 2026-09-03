from rapidfuzz import fuzz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("data/processed/cartwise_normalized.csv")

print("Dataset shape:", df.shape)

print("\nFirst 5 products:")
print(df[["platform", "product_name", "normalized_name"]].head())


vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(df["normalized_name"])

print("\nTF-IDF matrix shape:")
print(tfidf_matrix.shape)


similarity_matrix = cosine_similarity(tfidf_matrix)

print("\nSimilarity matrix shape:")
print(similarity_matrix.shape)


# -----------------------------
# Matching helper functions
# -----------------------------

def brand_match(brand_1, brand_2):
    if pd.isna(brand_1) or pd.isna(brand_2):
        return 0

    brand_1 = str(brand_1).strip().lower()
    brand_2 = str(brand_2).strip().lower()

    if not brand_1 or not brand_2:
        return 0

    return int(brand_1 == brand_2)

def quantity_match(row_1, row_2):
    if pd.isna(row_1["quantity_value"]) or pd.isna(row_2["quantity_value"]):
        return 0

    value_match = float(row_1["quantity_value"]) == float(row_2["quantity_value"])
    unit_match = str(row_1["quantity_unit"]).lower() == str(row_2["quantity_unit"]).lower()
    pack_match = float(row_1["pack_count"]) == float(row_2["pack_count"])

    if value_match and unit_match and pack_match:
        return 1

    return -1

def fuzzy_similarity(name_1, name_2):
    return fuzz.ratio(name_1, name_2) / 100
# -----------------------------
# Generate matches
# -----------------------------

matches = []

for i in range(len(df)):
    for j in range(i + 1, len(df)):

        if df.iloc[i]["platform"] == df.iloc[j]["platform"]:
            continue

        name_similarity = similarity_matrix[i][j]
        fuzzy_score = fuzzy_similarity(
    df.iloc[i]["normalized_name"],
    df.iloc[j]["normalized_name"]
)

        brand_score = brand_match(
            df.iloc[i]["brand"],
            df.iloc[j]["brand"]
        )

        quantity_score = quantity_match(
            df.iloc[i],
            df.iloc[j]
        )

        final_score = (
    0.40 * name_similarity +
    0.20 * fuzzy_score +
    0.20 * brand_score +
    0.20 * quantity_score
)
        if final_score >= 0.50:
            print(
        f"{df.iloc[i]['product_name']} | "
        f"{df.iloc[i]['platform']}  <->  "
        f"{df.iloc[j]['product_name']} | "
        f"{df.iloc[j]['platform']}  | "
        f"score={final_score:.3f}"
    )

        matches.append({
            "product_1": df.iloc[i]["product_name"],
            "platform_1": df.iloc[i]["platform"],
            "product_2": df.iloc[j]["product_name"],
            "platform_2": df.iloc[j]["platform"],
            "name_similarity": name_similarity,
            "fuzzy_similarity": fuzzy_score,
            "brand_match": brand_score,
            "quantity_match": quantity_score,
            "final_score": final_score
        })


matches_df = pd.DataFrame(matches)

matches_df = matches_df.sort_values(
    by="final_score",
    ascending=False
)

print("\nTop 10 entity matches:")
print(matches_df.head(10).to_string(index=False))

# -----------------------------
# Evaluate labelled pairs
# -----------------------------

eval_df = pd.read_csv("data/processed/entity_matching_eval.csv")

correct = 0
total = 0

print("\nEvaluation results:")

for _, row in eval_df.iterrows():

    product_1 = df[
        (df["product_name"].str.strip().str.lower() == row["product_1"].strip().lower()) &
        (df["platform"].str.strip().str.lower() == row["platform_1"].strip().lower())
    ]

    product_2 = df[
        (df["product_name"].str.strip().str.lower() == row["product_2"].strip().lower()) &
        (df["platform"].str.strip().str.lower() == row["platform_2"].strip().lower())
    ]

    # Skip pairs that cannot be found
    if product_1.empty or product_2.empty:
        continue

    p1 = product_1.iloc[0]
    p2 = product_2.iloc[0]

    # Calculate name similarity
    name_score = cosine_similarity(
        vectorizer.transform([p1["normalized_name"]]),
        vectorizer.transform([p2["normalized_name"]])
    )[0][0]

    # Calculate brand and quantity signals
    brand_score = brand_match(
        p1["brand"],
        p2["brand"]
    )

    quantity_score = quantity_match(
        p1,
        p2
    )

    # Calculate final score
    final_score = (
        0.60 * name_score +
        0.20 * brand_score +
        0.20 * quantity_score
    )

    prediction = int(final_score >= 0.70)
    actual = int(row["label"])

    if prediction == actual:
        correct += 1

    total += 1

    print(
        f"{p1['product_name']} <-> "
        f"{p2['product_name']} | "
        f"score={final_score:.3f} | "
        f"actual={actual} | "
        f"predicted={prediction}"
    )


if total > 0:

    accuracy = correct / total

    print("\nEvaluation summary:")
    print("Pairs evaluated:", total)
    print("Correct predictions:", correct)
    print(f"Accuracy: {accuracy:.2%}")

else:
    print("\nNo valid cross-platform evaluation pairs found.")