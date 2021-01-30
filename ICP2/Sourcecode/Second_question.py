num=int(input('Enter a number:\n'))
steps=0
if num>0:
    while num>0 :
        steps=steps+1
        if num%2==0:
            num=num/2
        else:
            num=num-1
print('The number of steps is',steps)
