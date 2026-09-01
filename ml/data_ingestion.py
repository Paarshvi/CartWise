import os
import json
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv


# Load .env
load_dotenv()

API_KEY = os.getenv("QUICKCOMMERCE_API_KEY")

API_URL = "https://api.quickcommerceapi.com/v1/search"


def search_products(query, latitude, longitude, pincode, platform):
    """Fetch products from the QuickCommerce API."""

    headers = {
        "X-API-Key": API_KEY
    }

    params = {
        "q": query,
        "lat": latitude,
        "lon": longitude,
        "platform": platform,
        "pincode": pincode
    }

    response = requests.get(
        API_URL,
        headers=headers,
        params=params
    )

    response.raise_for_status()

    return response.json()


def standardize_products(data, platform, pincode):
    """Convert API response into CartWise's standard format."""

    products = data["data"]["products"]

    standardized = []

    for product in products:

        standardized.append({
            "platform": platform,
            "pincode": pincode,
            "product_id": product["id"],
            "product_name": product["name"],
            "brand": product["brand"],
            "price": product["offer_price"],
            "mrp": product["mrp"],
            "quantity": product["quantity"],
            "available": product["available"],
            "inventory": product["inventory"],
            "eta": product["platform"]["sla"],
            "store_id": product["store_id"],
            "timestamp": datetime.now().isoformat()
        })

    return standardized


def save_products(products, filename):
    """Save standardized products as CSV."""

    df = pd.DataFrame(products)

    df.to_csv(filename, index=False)

    print(f"Saved {len(df)} products to {filename}")


if __name__ == "__main__":

    # Location
    latitude = 28.715345
    longitude = 77.108801
    pincode = "110085"

    # Search
    query = "milk"
    platform = "Zepto"

    # Get API data
    data = search_products(
        query,
        latitude,
        longitude,
        pincode,
        platform
    )

    # Standardize
    products = standardize_products(
        data,
        platform,
        pincode
    )

    # Save
    save_products(
    products,
    "data/processed/zepto_milk.csv"
)

    # Show result
    print("\nFirst 5 standardized products:\n")

    print(
        pd.DataFrame(products)
        .head()
        .to_string(index=False)
    )