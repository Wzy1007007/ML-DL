# 手撕梯度下降、LR、交叉熵、Softmax

### 梯度下降

```
import numpy as np

def compute_loss(w, b, x, y):
	"""
	计算二次损失函数
	
	参数：
	w (float): 权重
	b (float): 偏置
	x (float): 输入值
	y (float): 真实标签
	
	返回：
	float: 损失值
	"""
	y_pred = w * x + b
	loss = 0.5 * (y - y_pred) ** 2  # 二次损失
	return loss

def compute_gradients(w, b, x, y):
	"""
	计算损失函数对w和b的梯度
	参数：
	w (float): 权重
	b (float): 偏置
	x (float): 输入值
	y (float): 真实标签
	
	返回：
	tuple: 返回梯度（dw, db）
	"""
	y_pred = w * x + b
	dw = -(y - y_pred) * x  # 对w的梯度
	db = -(y - y_pred)      # 对b的梯度
	return dw, db

def gradient_descent(x, y, learning_rate=0.1, num_iterations=100):
    """
    基于梯度下降法优化w和b
    参数：
    x (array): 输入值数组
    y (array): 真实标签数组
    w_init (float): 初始权重
    b_init (float): 初始偏置
    learning_rate (float): 学习率
    num_iterations (int): 迭代次数

    返回：
    tuple: 最终的优化参数w和b
    """
    w = 0
    b = 0
    loss_history = []

    for i in range(num_iterations):
        dw, db = 0, 0
        total_loss = 0  # 用于累加所有样本的损失值
        # 计算所有样本的梯度
        for j in range(len(x)):
            dw_i, db_i = compute_gradients(w, b, x[j], y[j])
            dw += dw_i
            db += db_i
            total_loss += compute_loss(w, b, x[j], y[j])  # 累加损失值

        # 求平均梯度
        dw /= len(x)
        db /= len(x)

        # 更新权重和偏置
        w -= learning_rate * dw
        b -= learning_rate * db

        # 计算平均损失并记录
        average_loss = total_loss / len(x)  # 计算平均损失
        loss_history.append(average_loss)

        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {average_loss:.4f}, w: {w:.4f}, b: {b:.4f}")

    return w, b, loss_history

# 示例数据（简单的线性回归问题）

x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([3, 5, 7, 9, 11])  # y = 2 * x + 1

# 使用梯度下降优化w和b

w_final, b_final, loss_history = gradient_descent(x_data, y_data, learning_rate=0.1, num_iterations=100)

print(f"最终的权重: w = {w_final:.4f}, 偏置: b = {b_final:.4f}")
```

### LR

```
import numpy as np

# Sigmoid函数

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

# 损失函数：交叉熵

def compute_loss(y, y_hat):
	m = len(y)
	loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / m
	return loss

# 梯度下降实现

def logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
	m, n = X.shape  # m是样本数，n是特征数
	w = np.zeros(n)
	b = 0
	loss_history = []
	
	# 梯度下降
	for i in range(num_iterations):
	    # 计算线性模型输出
	    z = np.dot(X, w) + b
	    y_hat = sigmoid(z)
	
	    # 计算损失
	    loss = compute_loss(y, y_hat)
	    loss_history.append(loss)
	
	    # 计算梯度
	    dw = np.dot(X.T, (y_hat - y)) / m
	    db = np.sum(y_hat - y) / m
	
	    # 更新参数
	    w -= learning_rate * dw
	    b -= learning_rate * db
	
	    # 每100次迭代输出一次损失
	    if i % 100 == 0:
	        print(f"Iteration {i}, Loss: {loss:.4f}")
	
	return w, b, loss_history

# 示例数据（简单二分类问题）

X = np.array([[1, 2], [1, 3], [2, 3], [4, 5], [6, 7]])  # 输入特征
y = np.array([0, 0, 0, 1, 1])  # 真实标签

# 训练逻辑回归模型

w, b, loss_history = logistic_regression(X, y, learning_rate=0.1, num_iterations=1000)

print("训练完成的权重:", w)
print("训练完成的偏置:", b)
```

### 交叉熵

```python
# 二分类交叉熵
import numpy as np

def binary_cross_entropy(y_true, y_pred):
    """
    计算二分类交叉熵损失
    
    参数：
    y_true (numpy array): 真实标签，0 或 1
    y_pred (numpy array): 模型预测的概率（预测为正类的概率）
    
    返回：
    float: 计算得到的交叉熵损失
    """
    # 防止log(0)，对预测概率做裁剪
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    
    # 交叉熵公式： - [y * log(p) + (1 - y) * log(1 - p)]
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

# 示例：真实标签和预测概率
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])

# 计算交叉熵损失
loss = binary_cross_entropy(y_true, y_pred)
print(f"Binary Cross-Entropy Loss: {loss}")

# 多分类交叉熵
import numpy as np
def categorical_cross_entropy(y_true, y_pred):
    """
    计算多分类交叉熵损失
    
    参数：
    y_true (numpy array): 真实标签的one-hot编码（例如 [0, 1, 0]）
    y_pred (numpy array): 模型预测的类别概率分布
    
    返回：
    float: 计算得到的交叉熵损失
    """
    # 防止log(0)，对预测概率做裁剪
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    
    # 交叉熵公式： - sum(p * log(q))
    loss = - np.sum(y_true * np.log(y_pred), axis=1)
    return np.mean(loss)

# 示例：真实标签和预测概率（one-hot编码）
y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
y_pred = np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.1, 0.8], [0.3, 0.6, 0.1], [0.8, 0.1, 0.1]])

# 计算交叉熵损失
loss = categorical_cross_entropy(y_true, y_pred)
print(f"Categorical Cross-Entropy Loss: {loss}")
```

### Softmax

```
import numpy as np

def softmax(z):
"""
计算Softmax函数
参数：
z (numpy array): 输入向量
返回：
numpy array: Softmax后的概率分布
"""
	exp_z = np.exp(z - np.max(z))  # 防止溢出，减去最大值
	return exp_z / np.sum(exp_z)

def softmax_derivative(z):
"""
计算Softmax的导数
参数：
z (numpy array): 输入向量

返回：
numpy array: Softmax导数矩阵
"""
	p = softmax(z)
	S = np.diag(p) - np.outer(p, p)  # 雅可比矩阵
	return S
	
# 示例

z = np.array([2.0, 1.0, 0.1])  # 模型的输出（未归一化的得分）
softmax_output = softmax(z)
softmax_derivative_output = softmax_derivative(z)

print("Softmax Output:", softmax_output)
print("Softmax Derivative Matrix:\n", softmax_derivative_output)
```