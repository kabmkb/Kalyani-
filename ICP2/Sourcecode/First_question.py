list = []            #Define empty list
finallist = []     #Define final list
N = int(input("Enter number of students : "))       #use input function to input N
print('Enter the height of students in feet')
for i in range(0, N):                               #Start for loop ranging from 0 to N
    student = float(input())                        #input student input data in float
    list.append(student)                            #append students height in the list

for i in range(0, N) :                              #iterating to the list, calculate the height of students in cm, and append to  the final list
        student = list[i]
        student = round(student * 30.48, 2)
        finallist.append(student)
print('The height of the students in centimeters is')
print(finallist)                                    #print the final list in cm
