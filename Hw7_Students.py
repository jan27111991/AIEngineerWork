class Student:
    # Init Method or Constructor
    def __init__(self, name, grade, department):  # Fixed parameter list
        self.name = name
        self.grade = grade
        self.department = department

    # Method to print student details
    def print_info(self):
        print("Student name is ", self.name)
        print("Student Grade is ", self.grade)
        print("Student Department is ", self.department)

    def update_grade(self, new_grade):
        self.grade = new_grade
        print(f"Updated {self.name}'s grade to {new_grade}")

if __name__ == "__main__":
    # Create student objects
    students = [
        Student("Alice Johnson", "A", "Computer Science"),
        Student("Bob Smith", "B", "Mathematics"),
        Student("Carol Williams", "A+", "Physics")
    ]
   
    print("Student Details:")
    for student in students:
        student.print_info()
        print("-----")  

    # Update grade for a student
    students[1].update_grade("A-")
    students[1].print_info()

    # Display students separately
    print("Individual Student Records:")

    for i, student in enumerate(students, 1):
        print(f"Student {i}:")
        student.print_info()