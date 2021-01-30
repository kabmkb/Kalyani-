list = []
finallist = []
N = int(input("Enter number of students : "))
print('Enter the height of students in feet')
for i in range(0, N):
    student = float(input())
    list.append(student)

for i in range(0, N) :
        student = list[i]
        student = round(student * 30.48, 2)
        finallist.append(student)
print('The height of the students in centimeters is')
print(finallist)
