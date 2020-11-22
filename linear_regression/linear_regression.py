import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
data = pd.read_csv("./Dataset/mlm.csv")
'''
#############################sklearn##########################
x, y, z = [], [], []
for i in range(0, 800):
    x.append([data.values[i][0], data.values[i][1],1.0])
    z.append(data.values[i][2])
# print(x)
x_test, y_test, z_test = [], [], []
for i in range(800, 999):
    x_test.append(data.values[i][0])
    y_test.append(data.values[i][1])
    z_test.append(data.values[i][2])

Linear = linear_model.LinearRegression()
Linear.fit(x, z)
ax = plt.axes(projection="3d")
ax.scatter3D(x_test, y_test, z_test)
x_drawing = np.linspace(0, 100)
y_drawing = np.linspace(0, 100)
X_drawing, Y_drawing = np.meshgrid(x_drawing, y_drawing)
ax.plot_surface(X=X_drawing,Y=Y_drawing,Z=X_drawing * Linear.coef_[0] + Y_drawing * Linear.coef_[1] + Linear.intercept_,color='b',alpha=0.3)

ax.view_init(elev=30, azim=30)
plt.show()
'''
###############################矩阵方法################################
x1,x2,y=[],[],[]
for i in range(0, 800):
    x1.append(data.values[i][0])
    x2.append(data.values[i][1])
    y.append(data.values[i][2])
x1=np.array(x1).reshape(800,1)
x2=np.array(x2).reshape(800,1)
y=np.array(y).reshape(800,1)
x=np.hstack((x1,x2,np.ones((800,1),dtype=int)))
# print(x)
w_ = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(x),x)), np.transpose(x)) ,y)
y_=np.dot(x,w_)
E = np.sum(pow((y - y_), 2))/len(x1)
print('E:', E)
x_test, y_test, z_test = [], [], []
for i in range(800, 999):
    x_test.append(data.values[i][0])
    y_test.append(data.values[i][1])
    z_test.append(data.values[i][2])
ax = plt.axes(projection="3d")
ax.scatter3D(x_test, y_test, z_test)
x_drawing = np.linspace(0, 100)
y_drawing = np.linspace(0, 100)
X_drawing, Y_drawing = np.meshgrid(x_drawing, y_drawing)
ax.plot_surface(X=X_drawing,Y=Y_drawing,Z=X_drawing * w_[0] + Y_drawing * w_[1] + w_[2],color='g',alpha=0.3)
ax.view_init(elev=30, azim=30)
plt.show()
