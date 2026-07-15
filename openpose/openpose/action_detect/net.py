
from torch import nn, randn, exp, sum

class NetV1(nn.Module):

    def __init__(self):
        super().__init__()

        self.W = nn.Parameter(randn(16384,2))

    # 前项过程逻辑
    def forward(self, x):
       h = x@self.W
       # soft max
       h = exp(h)
       z = sum(h,dim=1,keepdim=True) #保持梯度
       return h/z

class NetV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Linear(16384,100),
            nn.ReLU(),
            nn.Linear(100,2),
            nn.Softmax(dim=1)
        )


    def forward(self, x):

        return self.sequential(x)

####添加
# 简单的三层全连接网络
class simpleNet(nn.Module):
    # simpleNet(28 * 28, 300, 100, 10)
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim,n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2,out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def get_name(self): # 返回类名
        return self.__class__.__name__


# 添加激活函数，增加网络的非线性
class activationNet(nn.Module):
    # activationNet(28 * 28, 300, 100, 10)
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(activationNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        # 注: 最后一层输出层不能添加激活函数，因为输出的结果为实际的类别

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def get_name(self):
        return self.__class__.__name__


# 添加批标准化，加快收敛速度
class batchNet(nn.Module):
    # batchNet(28 * 28, 300, 100, 10)
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(batchNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2),
            nn.ReLU(True),
        )
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        # 注：批标准化一般放在全连接层的后面，非线性层(激活函数)的前面

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def get_name(self):
        return self.__class__.__name__
