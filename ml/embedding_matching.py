import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load normalized product data
df = pd.read_csv("data/processed/cartwise_normalized.csv")

print("Dataset shape:", df.shape)


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# Create embeddings for product names
embeddings = model.encode(
    df["normalized_name"].tolist(),
    show_progress_bar=True
)

print("Embedding shape:", embeddings.shape)


# Load evaluation pairs
eval_df = pd.read_csv("data/processed/entity_matching_eval.csv")


print("\nEmbedding evaluation:")

correct = 0

for _, row in eval_df.iterrows():

    product_1 = df[
        (df["product_name"].str.strip().str.lower() == row["product_1"].strip().lower()) &
        (df["platform"].str.strip().str.lower() == row["platform_1"].strip().lower())
    ]

    product_2 = df[
        (df["product_name"].str.strip().str.lower() == row["product_2"].strip().lower()) &
        (df["platform"].str.strip().str.lower() == row["platform_2"].strip().lower())
    ]

    if product_1.empty or product_2.empty:
        continue

    index_1 = product_1.index[0]
    index_2 = product_2.index[0]

    score = cosine_similarity(
        embeddings[index_1].reshape(1, -1),
        embeddings[index_2].reshape(1, -1)
    )[0][0]

    prediction = int(score >= 0.70)

    if prediction == row["label"]:
        correct += 1

    print(
        f"{row['product_1']} <-> {row['product_2']} "
        f"| score={score:.3f} "
        f"| actual={row['label']} "
        f"| predicted={prediction}"
    )


print("\nEmbedding evaluation summary:")
print("Pairs evaluated:", len(eval_df))
print("Correct predictions:", correct)
print(f"Accuracy: {correct / len(eval_df) * 100:.2f}%")