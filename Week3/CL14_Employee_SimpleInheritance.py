class Employee:
    def __init__(self, name, employee_id):
        self.name = name
        self.employee_id = employee_id

    def display_info(self):
        print(f"Employee Name: {self.name}, Employee ID: {self.employee_id}")

class Tester(Employee):
    def __init__(self, name, employee_id, testing_tools):
        super().__init__(name, employee_id)
        self.testing_tools = testing_tools

    def run_tests(self):
        print(f"{self.name} is running automated tests using {self.testing_tools}.")

p1= Tester("Alice", "T123", ["Selenium", "PyTest"])
p1.run_tests()
p1.display_info()   
"""
if __name__ == "__main__":
    tester1 = Tester("Alice", "T123", ["Selenium", "PyTest"])
    tester1.display_info()      # Parent method
    tester1.run_tests()         # Child method

    tester2 = Tester("Alice2", "T123", ["Playwright"])
    tester2.display_info()      # Parent method
    tester2.run_tests()         # Child method
"""