import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#=====================
# 1. 数据读取函数
#=====================
    @@ -32,21 +33,30 @@ def replace_nan_with_mean(X):
#=====================
# 读取训练集

X_train, y_train = load_dataset(r"C:\Users\戴怡广\Desktop\test4\logsitic\horseColicTraining.txt")
# 读取测试集
X_test, y_test = load_dataset(filename)


#=====================
# 4. 构建并训练逻辑回归模型
#=====================
X_train = replace_nan_with_mean(X_train)
X_test = replace_nan_with_mean(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)




#=====================
# 5. 测试集预测
#=====================
y_pred = model.predict(X_test)


#=====================
# 6. 计算准确率
#=====================
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")
