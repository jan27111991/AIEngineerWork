import pandas as pd
def total_revenue(data):
    df = pd.DataFrame(data)
    df['Total_Revenue'] = df['Quantity'] * df['Price']
    total_revenue = df['Total_Revenue'].sum()
    return total_revenue
# Sample data
data = {
    "Product": ["Pen", "Book", "Bag", "Laptop"],
    "Quantity": [10, 5, 2, 1],
    "Price": [20, 200, 600, 1500]
}
result = total_revenue(data)
print(result)