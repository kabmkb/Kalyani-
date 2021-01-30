num=int(input('Enter a number:\n'))                 #input a integer
steps=0                                             #intialising step variable to 0
if num>0:                                           #checking if the number is positive
    while num!=0 :                                  # while num is not equals to zero execute the loop
        steps=steps+1                               #increment steps
        if num%2==0:                                #checking if the number is even
            num=num/2
        else:
            num=num-1s
print('The number of steps is',steps)
