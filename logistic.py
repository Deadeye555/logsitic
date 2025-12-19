import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 请确认文件路径正确，如果报错找不到文件，请检查路径或将文件放在同级目录下
filename = r"c:\Users\Administrator\Desktop\机器学习\lesson4\testSet.txt"

#=====================
# 1. 数据读取函数
#=====================
def load_dataset(filename):
    try:
        data = np.loadtxt(filename)
        X = data[:, :-1]   # 取除最后一列外的所有列作为特征
        y = data[:, -1]    # 取最后一列作为标签
        return X, y
    except Exception as e:
        print(f"读取文件出错: {e}")
        return None, None

#=====================
# 2. 缺失值处理函数
#   （假设 0 代表缺失值，将其替换为非0数值的均值）
#=====================
def replace_nan_with_mean(X):
    # 为了不影响原始数据，建议创建一个副本
    X_copy = X.copy()
    m, n = X_copy.shape
    for i in range(n):
        col = X_copy[:, i]
        # 选择非0的数作为有效特征
        valid = col[col != 0]
        if len(valid) > 0:
            mean_val = np.mean(valid)
            # 将该列中为0的位置替换为均值
            col[col == 0] = mean_val
            X_copy[:, i] = col
    return X_copy

#=====================
# 3. 主流程
#=====================
if __name__ == "__main__":
    # A. 读取数据
    print("正在读取数据...")
    X, y = load_dataset(filename)

    if X is not None:
        # B. 数据预处理（处理缺失值）
        X = replace_nan_with_mean(X)
        
        # C. 划分训练集和测试集 (80% 训练, 20% 测试)
        # random_state=42 保证每次运行结果一致
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

        #=====================
        # 4. 构建并训练逻辑回归模型
        #=====================
        print("正在训练模型...")
        # max_iter 适当调大以防止不收敛警告
        model = LogisticRegression(max_iter=1000) 
        model.fit(X_train, y_train)

        #=====================
        # 5. 测试集预测
        #=====================
        y_pred = model.predict(X_test)
        
        #=====================
        # 6. 计算准确率
        #=====================
        accuracy = model.score(X_test, y_test)
        print("="*30)
        print(f"模型准确率 (Accuracy): {accuracy * 100:.2f}%")
        print("="*30)
        
        # 可选：打印前10个预测结果对比
        print("前10个样本预测对比 (预测值 vs 真实值):")
        print(np.column_stack((y_pred[:10], y_test[:10])))
