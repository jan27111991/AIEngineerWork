class Employee:
    def __init__(self, name):
        self.name = name
        
    def display_info(self):
        print(f"Person Name: {self.name}")

class AutomationSkills:
    def write_scripts(self):
        print("Writing Selenium scripts")

class AutomationTester(Employee, AutomationSkills):
    def execute_Tests(self):
        print(f"Automation Tester Name: {self.name}")

p1 = AutomationTester("Alice")
p1.display_info() 
p1.write_scripts()        
p1.execute_Tests()