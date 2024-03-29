import os
import pandas as pd
import torch

os.makedirs(os.path.join('D:\project\d2l','data'),exist_ok=True)
data_file = os.path.join('D:\project\d2l','data','house_tiny.csv')

with open(data_file,'w') as f:            #with 语句是一种上下文管理器，用于确保文件在使用完毕后会被正确关闭.
    f.write('NumRooms,Alley,Price\n')     #即使发生异常。这有助于避免资源泄漏和确保代码的可靠性
    f.write('NA,Pave,127500\n')           #文件对象 f 是在 with 语句块内的一个局部变量，你可以使用它执行文件操作。
    f.write('2,NA,10600\n')               #当 with 语句块结束时，文件会被自动关闭，不需要显式调用 f.close()。这确保了文件在使用完毕后会被正确关闭，释放系统资源。
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]
inputs = inputs.fillna(inputs.mean(numeric_only=True),axis=0)                #.fillna() 是 Pandas 中用于填充缺失值（NaN）的方法。
print(inputs)                                                                #缺失值是数据集中的空值或未知值，通常表示为 NaN（Not a Number）。
                                                                             #numeric_only：（可选参数）如果设置为 True，则仅计算数字列的平均值

inputs = pd.get_dummies(inputs,dummy_na=True)                                #dummy_na：（可选参数）是否为缺失值创建一个独热编码列，默认为 False
print(inputs)

X = torch.tensor(inputs.to_numpy(dtype=float))
Y = torch.tensor(outputs.to_numpy(dtype=float))
print(X,Y)