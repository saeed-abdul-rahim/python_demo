import numpy as np

a = [1,2,3,4,5]
b = [2,3,4,5,6]

ab = np.dot(a,b)
print(ab)

c = [1,2]
d = [[2,4,6],[3,5,7]]
e = [[8],[9],[10]]
cd = np.dot(c,d)
de = np.dot(d,e)
print(cd,de)