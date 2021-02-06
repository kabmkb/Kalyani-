#Third program
# importing numpy library
import numpy as np
a=np.random.randint(20, size=(4,5) )        #creating random vector of range 20 and reshaping it to the 4X5 size
print(a)
a[np.where(a==np.max(a,axis=1,keepdims=True))]=0    #replacing the max no. in a row by 0
print(a)