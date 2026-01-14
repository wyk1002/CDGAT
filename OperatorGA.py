import numpy as np 

def OperatorGA(Parent,lower,upper):
    proC = 1
    disC = 20
    proM = 1
    disM = 20

    Parent    = Parent.decs
    #print("Parent.shape",Parent.shape)
    Parent1 = Parent[:int(np.floor(len(Parent)/2)), :]
    Parent2 = Parent[int(np.floor(len(Parent)/2)):int(np.floor(len(Parent)/2))*2, :]
    #print("Parent1.shape:",Parent1.shape)
    #print("Parent1:",Parent1)
    #print("Parent2.shape:",Parent2.shape)
    #print("Parent2:",Parent2)
    Offspring= GAreal(Parent1,Parent2,lower,upper,proC,disC,proM,disM)

    return Offspring

def GAreal(Parent1,Parent2,lower,upper,proC,disC,proM,disM):
    #print("Parent1.shape",Parent1.shape)
    N, D = Parent1.shape

    # 初始化 beta 和 mu
    beta = np.zeros((N, D))
    mu = np.random.rand(N, D)   #这里的随机化的取值范围是[0,1) #变化
    #a=np.zeros((int(N/2),D))
    #b=np.ones((int(N/2),D))
    #mu = np.concatenate((a,b),axis=0)

    # 计算 beta 值
    beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (disC + 1))
    beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (disC + 1))

    # 对 beta 进行随机变化
    beta *= (-1) ** np.random.randint(2, size=(N, D))#变化
    #beta *= (-1) ** np.concatenate((a,b),axis=0)
    # 随机将部分 beta 设置为 1
    beta[np.random.rand(N, D) < 0.5] = 1 #变化
    #beta[np.concatenate((a,b),axis=0) < 0.5] = 1
    # 随机将一部分 beta 设置为 1
    beta[np.tile(np.random.rand(N, 1) > proC, (1, D))] = 1 
    
    # 根据公式计算 Offspring
    Offspring = np.concatenate([(Parent1 + Parent2) / 2 + beta * (Parent1 - Parent2) / 2,
                                (Parent1 + Parent2) / 2 - beta * (Parent1 - Parent2) / 2], axis=0)
    #print("Offspring.shape",Offspring.shape)
    #print("Offspring[1199]",Offspring[1199])
    #print("Offspring[1200]",Offspring[1200])
    #print("Offspring",Offspring)
    # 重复 lower 和 upper
    Lower = repmat(lower, 2 * N, 1)
    Upper = repmat(upper, 2 * N, 1)

    # 生成 Site 和 mu
    Site = np.random.rand(2 * N, D) < (proM / D)  #变化
    #Site = np.concatenate((a,b,a,b),axis=0) < (proM / D)
    mu = np.random.rand(2 * N, D)  #变化
    #mu = np.concatenate((b,a,b,a),axis=0)

    # 计算 Offspring
    temp = np.logical_and(Site, mu <= 0.5)
    Offspring = np.minimum(np.maximum(Offspring, Lower), Upper)
    Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
            (2 * mu[temp] + (1 - 2 * mu[temp]) *
            (1 - (Offspring[temp] - Lower[temp]) / (Upper[temp] - Lower[temp])) ** (disM + 1)) **
            (1 / (disM + 1)) - 1)

    temp = np.logical_and(Site, mu > 0.5)
    Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
            1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) *
                (1 - (Upper[temp] - Offspring[temp]) / (Upper[temp] - Lower[temp])) ** (disM + 1)) **
            (1 / (disM + 1)))

    return Offspring

def repmat(A, m, n):
    return np.tile(A, (m, n))