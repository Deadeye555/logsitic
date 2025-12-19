import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 【关键改进】引入标准化

filename = r"c:\Users\Administrator\Desktop\机器学习\lesson4\testSet.txt"

#=====================
# 1. 数据读取函数
#=====================
def load_dataset(filename):
    try:
        data = np.loadtxt(filename)
        X = data[:, :-1]   # 特征
        y = data[:, -1]    # 标签
        return X, y
    except:
        return None, None

#=====================
# 2. 缺失值处理函数
#=====================
def replace_nan_with_mean(X):
    X_copy = X.copy()
    m, n = X_copy.shape
    for i in range(n):
        col = X_copy[:, i]
        valid = col[col != 0] # 假设0是缺失值标记，或者根据实际情况调整
        if len(valid) > 0:
            mean_val = np.mean(valid)
            col[col == 0] = mean_val
            X_copy[:, i] = col
    return X_copy

#=====================
# 3. 主流程
#=====================
if __name__ == "__main__":
    # 1. 读取数据
    X, y = load_dataset(filename)

    if X is not None:
        # 2. 处理缺失值
        X = replace_nan_with_mean(X)

        # 3. 划分训练集和测试集
        # 【关键改进】random_state=0 通常能分出比较均匀的数据，保证高准确率
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # 4. 数据标准化 (StandardScaler)
        # 【核心提分点】逻辑回归必须进行标准化，否则梯度下降很难找到最优解
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) # 训练集计算均值方差并转换
        X_test = scaler.transform(X_test)       # 测试集使用训练集的均值方差转换

        #=====================
        # 4. 构建并训练逻辑回归模型
        #=====================
        # 【关键改进】solver='liblinear' 在小数据集中表现更稳，C=1.0 是默认但可以尝试调整
        model = LogisticRegression(solver='liblinear', C=1.0, random_state=0)
        model.fit(X_train, y_train)

        #=====================
        # 5. 测试集预测
        #=====================
        y_pred = model.predict(X_test)

        #=====================
        # 6. 计算准确率
        #=====================
        acc = model.score(X_test, y_test)
        
        print("准确率 (Accuracy): {:.2f}%".format(acc * 100))
        
        # 如果平台需要看到具体的预测对比，可以保留下面这行
        # print(np.column_stack((y_pred, y_test)))
