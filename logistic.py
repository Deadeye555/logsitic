import numpy as np
from sklearn.linear_model import LogisticRegression




#=====================
# 1. 数据读取函数
#=====================


# 3. 主流程

#=====================
# 读取训练集
X_train, y_train = load_dataset(r"C:\Users\E507\Desktop\logsitic\horseColicTraining.txt")

# 读取测试集
X_test, y_test = load_dataset(r"C:\Users\E507\Desktop\logsitic\horseColicTest.txt")


# 处理训练集缺失值
X_train = replace_nan_with_mean(X_train)
# 处理测试集缺失值
X_test = replace_nan_with_mean(X_test)


#=====================
# 4. 构建并训练逻辑回归模型
#=====================
model = LogisticRegression()
model.fit(X_train, y_train)


#=====================
# 5. 测试集预测
#=====================
y_pred = model.predict(X_test)


#=====================
# 6. 计算准确率
#=====================
accuracy = np.mean(y_pred == y_test)
print(f"测试集准确率: {accuracy:.4f}")
