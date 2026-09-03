import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz


# Load data
df = pd.read_csv("data/processed/cartwise_normalized.csv")

print("Dataset shape:", df.shape)


# -----------------------------
# TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    df["normalized_name"]
)


# -----------------------------
# Embeddings
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    df["normalized_name"].tolist(),
    show_progress_bar=True
)

print("Embedding shape:", embeddings.shape)


# -----------------------------
# Helper functions
# -----------------------------

def fuzzy_similarity(name_1, name_2):
    return fuzz.ratio(name_1, name_2) / 100


def brand_match(brand_1, brand_2):
    if pd.isna(brand_1) or pd.isna(brand_2):
        return 0

    brand_1 = str(brand_1).strip().lower()
    brand_2 = str(brand_2).strip().lower()

    if not brand_1 or not brand_2:
        return 0

    return int(brand_1 == brand_2)


def quantity_match(row_1, row_2):
    if (
        pd.isna(row_1["quantity_value"]) or
        pd.isna(row_2["quantity_value"]) or
        pd.isna(row_1["quantity_unit"]) or
        pd.isna(row_2["quantity_unit"]) or
        pd.isna(row_1["pack_count"]) or
        pd.isna(row_2["pack_count"])
    ):
        return 0

    value_match = (
        float(row_1["quantity_value"]) ==
        float(row_2["quantity_value"])
    )

    unit_match = (
        str(row_1["quantity_unit"]).lower() ==
        str(row_2["quantity_unit"]).lower()
    )

    pack_match = (
        float(row_1["pack_count"]) ==
        float(row_2["pack_count"])
    )

    if value_match and unit_match and pack_match:
        return 1

    return -1


# -----------------------------
# Evaluation
# -----------------------------

eval_df = pd.read_csv(
    "data/processed/entity_matching_eval.csv"
)

y_true = []
y_pred = []
correct = 0

print("\nHybrid evaluation:")

for _, row in eval_df.iterrows():

    product_1 = df[
        (df["product_name"].str.strip().str.lower() ==
         row["product_1"].strip().lower()) &
        (df["platform"].str.strip().str.lower() ==
         row["platform_1"].strip().lower())
    ]

    product_2 = df[
        (df["product_name"].str.strip().str.lower() ==
         row["product_2"].strip().lower()) &
        (df["platform"].str.strip().str.lower() ==
         row["platform_2"].strip().lower())
    ]

    if product_1.empty or product_2.empty:
        continue

    index_1 = product_1.index[0]
    index_2 = product_2.index[0]

    # Embedding similarity
    embedding_score = cosine_similarity(
        embeddings[index_1].reshape(1, -1),
        embeddings[index_2].reshape(1, -1)
    )[0][0]

    # TF-IDF similarity
    tfidf_score = cosine_similarity(
        tfidf_matrix[index_1],
        tfidf_matrix[index_2]
    )[0][0]

    # Fuzzy similarity
    fuzzy_score = fuzzy_similarity(
        df.iloc[index_1]["normalized_name"],
        df.iloc[index_2]["normalized_name"]
    )

    # Metadata signals
    brand_score = brand_match(
        df.iloc[index_1]["brand"],
        df.iloc[index_2]["brand"]
    )

    quantity_score = quantity_match(
        df.iloc[index_1],
        df.iloc[index_2]
    )

    # Hybrid score
    final_score = (
    0.60 * embedding_score +
    0.25 * tfidf_score +
    0.15 * fuzzy_score
)
    if quantity_score == -1:
        prediction = 0
    else:
        prediction = int(final_score >= 0.70)

    y_true.append(row["label"])
    y_pred.append(prediction)

    if prediction == row["label"]:
        correct += 1

    print(
        f"{row['product_1']} <-> {row['product_2']} "
        f"| embedding={embedding_score:.3f} "
        f"| tfidf={tfidf_score:.3f} "
        f"| fuzzy={fuzzy_score:.3f} "
        f"| brand={brand_score} "
        f"| quantity={quantity_score} "
        f"| final={final_score:.3f} "
        f"| actual={row['label']} "
        f"| predicted={prediction}"
    )


print("\nHybrid evaluation summary:")
print("Pairs evaluated:", len(eval_df))
print("Correct predictions:", correct)
print(f"Accuracy: {correct / len(eval_df) * 100:.2f}%")

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"Precision: {precision * 100:.2f}%") 
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")