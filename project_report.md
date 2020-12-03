###神经网络项目实验报告
1. **任务描述**：
    - **线性回归模型**
    &nbsp;&nbsp;&nbsp;&nbsp;给定含有1000条记录的数据集mlm.csv，其中每条记录均包含两个自变量x,y和一个因变量z，它们之间存在较为明显的线性关系。  
    &nbsp;&nbsp;&nbsp;&nbsp;*任务*：请对数据进行三维可视化分析，并训练出良好的线性回归模型。
    
    - **非线性多分类器**
    &nbsp;&nbsp;&nbsp;&nbsp;鸢尾花数据集iris.csv含有150条记录，每条记录包含萼片长度sepal length、萼片宽度sepal width、 花瓣长度petal length和花瓣宽度petal width四个数值型特征，以及它的所属类别class（可能为Iris-setosa,Iris-versicolor,Iris-virginica三者之一）。  
    &nbsp;&nbsp;&nbsp;&nbsp;*任务*：请利用该数据集训练出一个良好的非线性分类器。
2. **模型描述**：
    - **线性回归模型**
     &nbsp;&nbsp;&nbsp;&nbsp;本实验采用两种实现方法，一是正规方程法，原理公式如下:  
     w=np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(x),x)),np.transpose(x)) ,y)  
     &nbsp;&nbsp;&nbsp;&nbsp;第二种方法是Sklearn库函数线性回归方法。
    
    - **非线性多分类器**
    &nbsp;&nbsp;&nbsp;&nbsp;本实验采用Pytorch机器学习工具搭建BP神经网络训练Iris数据。
3. **运行结果**：
    - **线性回归模型——正规方程法结果**
     ![avatar](F:\Microsoft-ai-edu-main\project_easy\result_matrix.png)
   
    - **线性回归模型——sklearn方法结果**
    ![avatar](F:\Microsoft-ai-edu-main\project_easy\result_sklearn.png)
    - **鸢尾花分类结果**
    本实验按照28比例分配训练集和测试集，得到的准确率为0.98。
     ![avatar](F:\Microsoft-ai-edu-main\project_easy\accuracy.PNG)