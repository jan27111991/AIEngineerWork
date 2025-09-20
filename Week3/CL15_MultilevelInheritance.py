class Person:
    def __init__(self, name):
        self.name = name
        

    def display_info(self):
        print(f"Person Name: {self.name}")

class Employee(Person):
    def __init__(self, name, employee_id):
        super().__init__(name)
        self.employee_id = employee_id

class Manager(Employee):
    def __init__(self, name, employee_id,team_size):
        super().__init__(name,employee_id)
        self.team_size = team_size

    def show_details(self):
        print(f"Manager Name: {self.name}, Employee ID: {self.employee_id}, Team Size: {self.team_size}")               
if __name__ == "__main__":
    employee1 = Manager("Janani","12345","4")
    employee1.show_details()
    employee1.display_info()