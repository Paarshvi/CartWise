import json
import pandas as pd


# Load raw API response
with open("data/raw/zepto_eggs_sample.json", "r") as file:
    data = json.load(file)


# Extract products
products = data["data"]["products"]


# Convert each product into our standard format
cartwise_products = []

for product in products:
    cartwise_products.append({
        "platform": product["platform"]["name"],
        "product_name": product["name"],
        "brand": product["brand"],
        "price": product["offer_price"],
        "mrp": product["mrp"],
        "quantity": product["quantity"],
        "available": product["available"],
        "inventory": product["inventory"],
        "eta": product["platform"]["sla"],
        "store_id": product["store_id"]
    })


# Create a DataFrame
df = pd.DataFrame(cartwise_products)


# Display the standardized data
print("\nCartWise Standardized Data:\n")
print(df.to_string(index=False))