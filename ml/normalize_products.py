import re
import pandas as pd


def normalize_text(text):
    """Clean and standardize product text."""

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Standardize litre/liter variations
    text = text.replace("litres", "l")
    text = text.replace("liters", "l")
    text = text.replace("litre", "l")
    text = text.replace("liter", "l")

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_quantity(text):
    """Extract quantity, unit and pack count."""

    if pd.isna(text):
        return None, None, 1

    text = str(text).lower().strip()

    # 1. Multipack quantities
    # Examples: 2 x 250 ml, 3 x 500 ml, 2 x 1 l

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(ml|l|g|kg)",
        text
    )

    if match:
        pack_count = float(match.group(1))
        value = float(match.group(2))
        unit = match.group(3)

        if unit == "l":
            value *= 1000
            unit = "ml"

        elif unit == "kg":
            value *= 1000
            unit = "g"

        return value, unit, pack_count

    # 2. Normal weight / volume quantities
    # Examples: 500 ml, 1 l, 500 g, 5 kg

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(ml|l|g|kg)",
        text
    )

    if match:
        value = float(match.group(1))
        unit = match.group(2)

        if unit == "l":
            value *= 1000
            unit = "ml"

        elif unit == "kg":
            value *= 1000
            unit = "g"

        return value, unit, 1

    # 3. Count-based quantities
    # Examples: 15 sheets, 1 pc, 6 pcs, 1 set

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(pc|pcs|piece|pieces|sheet|sheets|set|sets)",
        text
    )

    if match:
        value = float(match.group(1))
        unit = match.group(2)

        return value, unit, 1

    return None, None, 1




def normalize_products(df):
    """Apply normalization to the product dataset."""

    df = df.copy()

    # Normalize product names
    df["normalized_name"] = (
        df["product_name"]
        .apply(normalize_text)
    )

    # Extract quantity information
    quantities = df["quantity"].apply(
        extract_quantity
    )

    df["quantity_value"] = quantities.apply(
        lambda x: x[0]
    )

    df["quantity_unit"] = quantities.apply(
        lambda x: x[1]
    )

    df["pack_count"] = quantities.apply(
        lambda x: x[2]
    )
    df["quantity_type"] = df["quantity_unit"].apply(
    lambda unit:
        "volume" if unit == "ml"
        else "weight" if unit == "g"
        else "count" if unit in [
            "pc",
            "pcs",
            "piece",
            "pieces",
            "sheet",
            "sheets",
            "set",
            "sets"
        ]
        else None
)

    # Remove products without names
    df = df[
        df["normalized_name"].str.len() > 0
    ]

    # Remove duplicate product listings
    df = df.drop_duplicates(
        subset=[
            "platform",
            "product_id"
        ]
    )

    return df


if __name__ == "__main__":

    import glob
    import os

    input_files = glob.glob("data/raw/*.xlsx")

    all_data = []

    for input_file in input_files:

        print(f"\nProcessing: {input_file}")

        df = pd.read_excel(input_file)

        # Rename platform-specific columns
        df = df.rename(columns={
            "Product ID": "product_id",
            "Platform": "platform",
            "Name": "product_name",
            "Brand": "brand",
            "MRP": "mrp",
            "Category": "category",
            "Subcategory": "subcategory",
            "Quantity": "quantity",
            "Updated at": "timestamp",
            "Image URL": "image_url",
            "Deeplink": "deeplink"
        })

        # Normalize
        normalized_df = normalize_products(df)

        all_data.append(normalized_df)

        print(
            f"Normalized {len(normalized_df)} products"
        )

    # Combine all platforms
    combined_df = pd.concat(
        all_data,
        ignore_index=True
    )

    print("\n===== COMBINED DATASET =====")

    print("Total products:", len(combined_df))

    print("\nProducts by platform:")
    print(
        combined_df["platform"].value_counts()
    )

    # Save combined dataset
    output_file = (
        "data/processed/"
        "cartwise_normalized.csv"
    )

    combined_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved combined dataset to {output_file}"
    )