#first program
#defining a class
class Employee:

#initializing the count to zero
    count=0

#initializing the total salary to 0
    total_salary=0

#The Employee class's __init__ method is called, and the self parameter will reference the e object
    def __init__(self, name, family, salary, department):
        self.name=name
        self.family=family
        self.salary=salary
        self.department=department

        #Appending the count
        Employee.count=Employee.count+1
        #Adding the total salaries ofo all the employees
        Employee.total_salary=Employee.total_salary+self.salary

# definig the show_details function and printing the details
    def show_details(self):
        print(self.name, self.family, self.salary, self.department)

#defining the function
    def get_avg_salary():
        return (Employee.total_salary / Employee.count)

#defining the inheritance class
class Fulltime_Employee(Employee):
    def info(self):
        print("This class is inherited the Employee class")

E1=Employee("Kevin", 4, 50000, "sales")
E2=Employee("Jessica", 2, 35000, "marketing")
E3=Fulltime_Employee("Lia", 5, 75000, "audit")

E1.show_details()
E2.show_details()
E3.show_details()

E3.info()

#printing the total count
print("Total no. of people = " ,Employee.count)

#priniting the average salary
print("Average salary = ", Employee.get_avg_salary())




