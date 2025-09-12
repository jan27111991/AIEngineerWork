import OnlineShopping

# Only positional 
print("Bill 5:", OnlineShopping.calculate_bill(100, 2)) 
# With custom tax 
print("Bill 6:", OnlineShopping.calculate_bill(500, 2, tax=0.1)) 
# With custom discount 
print("Bill 7:", OnlineShopping.calculate_bill(500, 2, discount=50)) 
# With custom tax and discount 
print("Bill 8:", OnlineShopping.calculate_bill(500, 2, tax=0.08, discount=100))