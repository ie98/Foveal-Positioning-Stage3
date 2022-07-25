import numpy as np
list = [[0,0.2,0.5,0.9,0.8],[0.2,0.7,0.9,0.4],[0.6,0.2,0.1,0],[0.1,0.9,0,0]]
# list = [0.1,0.9,0,0]
arr = np.array(list)
arr = np.random.random((3,3))
print(arr)
index = np.where(arr >= 0.5)

print(index)

sum = np.sum(arr[index[0],index[1]])


print(sum)