import pandas as pd
import numpy as np
words = ['all', 'random', 'words', 'spelled', 'accurately', 'and', 'precisely', 'for', 'testing']
numbers = [43,675,87,21,87,45,0,3,8]
numbers2 = [40,65,76,1,8,5,0,31,5]
a = ['a', 'b', 'c','d','e','f','g','h','i']
dic = {'Word_Name': pd.Series(words, index = a), 'Numbers': pd.Series(numbers, index = a)}
dt = pd.DataFrame(dic)
print(dt)
print(dt.loc['a'])
print(dt[dt['Numbers'] >= 30])
num = {'one': pd.Series(numbers, index = a), 'two': pd.Series(numbers2, index = a)}
numdt = pd.DataFrame(num)
print(numdt.apply(np.mean))
print(numdt.applymap(lambda x: x>= 1))
cal = {'Word_Name': pd.Series(words), 'one': pd.Series(numbers), 'two': pd.Series(numbers2)}
caldt = pd.DataFrame(cal)
caldt = caldt[caldt['one'] >= 1][caldt['two'] >= 1]
avgdt = caldt[['one', 'two']].apply(np.mean)
print(caldt)
print(avgdt)