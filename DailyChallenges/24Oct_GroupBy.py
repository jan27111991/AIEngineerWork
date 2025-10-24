import pandas as pd
def clean_missing_data(data):
    df = pd.DataFrame(data)
    df['Quanity'] = df['Quantity'].fillna(0)
    df['Price'] = df['Price'].fillna(0)
    df['Quanity'] = pd.to_numeric(df['Quantity'], errors = 'coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors= 'coerce')
    df['Total'] = df['Quantity'] * df['Price']
    return df.to_dict(orient='records')
# Example usage
data = {
    "Product": ["Pen", "Book", "Bag", "Laptop"],
    "Quantity": [10, None, 2, None],
    "Price": [20,200,None,1500]
}
cleaned_data = clean_missing_data(data)
print(cleaned_data)