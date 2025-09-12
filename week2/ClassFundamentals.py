class Person:
    # Init Method or Constructor
    def __init__(self,name,age):
        self.name = name
        self.age = age 

    # Method to display person details 
    def invite(self):
        print("Welcome", self.name)
        print("Your age is ", self.age)

    #Create object of the class
p1 = Person("John",36)
p2 = Person("Jane",25)

p1.invite()
p2.invite()
    
