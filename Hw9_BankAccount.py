class BankAccount:
    def __init__(self, account_holder, account_type, balance=0):  # Fixed parameter order
        self.account_holder = account_holder
        self.balance = balance
        self.account_type = account_type

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            print(f"Deposited {amount}. New balance is {self.balance}.")
        else:
            print("Deposit amount must be positive.")

    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            print(f"Withdrawn {amount}. New balance is {self.balance}.")
        else:
            print("Invalid withdrawal amount.")

    def display_balance(self):
        return self.balance

if __name__ == "__main__":
    account1 = BankAccount("John Doe", "Savings", 1000)  # Fixed argument order
    account2 = BankAccount("Jane Smith", "Checking", 500)

    print(f"{account1.account_holder}'s initial balance: {account1.display_balance()}")
    account1.deposit(200)
    account1.withdraw(150)
    print(f"{account1.account_holder}'s final balance: {account1.display_balance()}")

    print(f"{account2.account_holder}'s initial balance: {account2.display_balance()}")
    account2.deposit(300)
    account2.withdraw(1000)  # This should show an invalid withdrawal message
    print(f"{account2.account_holder}'s final balance: {account2.display_balance()}")          