from scipy.spatial import distance
import numpy as np
from scipy.spatial.distance import cdist
from pop_class import *
import math
from .non_domination_scd_sort import*

def EnvironmentalSelection(Population,Popsize):

    n = Population.length
    dist = distance.pdist(Population.decs)
    dist = distance.squareform(dist)
    
    max_vals = np.max(Population.decs, axis=0)
    min_vals = np.min(Population.decs, axis=0)
    ranges = max_vals - min_vals
    product = np.prod(ranges)
    feature_count = Population.decs.shape[1]

    #几何平均：0.2*（决策范围）^(1/d)
    V = 0.2 * product ** (1. / feature_count)


    #支配矩阵
    DominationX = np.zeros((n,n))

    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] > V:
                continue
            L1 = Population.getObj(i) < Population.getObj(j)
            L2 = Population.getObj(i) > Population.getObj(j)
            if np.all(L1 | (~L2)):
                DominationX[i, j] = 0
                DominationX[j, i] = 1
            elif np.all(L2 | (~L1)):
                DominationX[i, j] = 1
                DominationX[j, i] = 0

    #局部支配密度 或 局部邻域优势指标
    #当前个体在几何平均距离V空间内支配解的个数
    LocalC = np.zeros(n)
    for i in range(n):
        tmp = dist[i, :]
        index = tmp < V
        LocalC[i] = np.sum(DominationX[i, index]) / np.sum(index)
    #LocalC[i] 越大，表示个体 i 在其局部邻域中“支配能力越强”，可能是非支配前沿中的优质解


    dist = np.sort(dist,axis=0)

    CrowdDis = np.sum(dist[0:3, :], axis=0)

    LocalC=LocalC.reshape(1,len(LocalC))
    CrowdDis=CrowdDis.reshape(1,len(CrowdDis))
    temp=np.concatenate((LocalC.T, (-CrowdDis).T), axis=1)
    #print("temp",temp)

    #按temp[0]即localC优先排序，其次crowddist,从低到高排
    index = np.lexsort((temp[:, 1], temp[:, 0]))
    #print("index",index)
    #Population.printPop()
    Population = Population.sort_index(index)

    #print("Population")
    #Population.printPop()

    if Population.length > Popsize:
        Population = Population.select_index(0, Popsize)

    CrowdDis = Crowding(Population.decs) 
    #返回的CrowdDis是N-1邻居
    return Population,CrowdDis

def Crowding(Pop):

    N, _ = Pop.shape
    K = N - 1
    Z = np.min(Pop, axis=0)
    Zmax = np.max(Pop, axis=0)
    pop = (Pop - np.tile(Z, (N, 1))) / np.tile((Zmax - Z), (N, 1))
    distance = cdist(pop, pop)
    value = np.sort(distance, axis=1)
    CrowdDis = K / np.sum(1 / (value[:, 1:N]+1e-12), axis=1)

    return CrowdDis

def dist_sort(Population):
    n = Population.length
    dist = distance.pdist(Population.decs)
    dist = distance.squareform(dist)
    
    max_vals = np.max(Population.decs, axis=0)
    min_vals = np.min(Population.decs, axis=0)
    ranges = max_vals - min_vals
    product = np.prod(ranges)
    feature_count = Population.decs.shape[1]
    V = 0.2 * product ** (1. / feature_count)
    '''
    print("dist.shape",dist.shape)
    print("dist",dist)

    print("V.shape",V.shape)
    print("V",V)
    '''
    DominationX = np.zeros((n,n))

    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] > V:
                continue
            L1 = Population.getObj(i) < Population.getObj(j)
            L2 = Population.getObj(i) > Population.getObj(j)
            if np.all(L1 | (~L2)):
                DominationX[i, j] = 0
                DominationX[j, i] = 1
            elif np.all(L2 | (~L1)):
                DominationX[i, j] = 1
                DominationX[j, i] = 0
    '''
    print("DominationX",DominationX)
    count_num=0
    for i in range(n):
        for j in range(n):
            if(DominationX[i, j]==1.0):
                count_num=count_num+1
                print("count_num",count_num)
                print(i)
                print(j)
    print("count_num",count_num)
    '''
    LocalC = np.zeros(n)
    for i in range(n):
        tmp = dist[i, :]
        index = tmp < V
        LocalC[i] = np.sum(DominationX[i, index]) / np.sum(index)

    #print("LocalC",LocalC)

    dist = np.sort(distance.squareform(distance.pdist(Population.decs)),axis=0)
    #print("dist.shape",dist.shape)
    #print("dist",dist)
    CrowdDis = np.sum(dist[0:3, :], axis=0)
    #print("CrowdDis.shape",CrowdDis.shape)
    #print("CrowdDis",CrowdDis)
    LocalC=LocalC.reshape(1,len(LocalC))
    CrowdDis=CrowdDis.reshape(1,len(CrowdDis))
    temp=np.concatenate((LocalC.T, (-CrowdDis).T), axis=1)
    #print("temp",temp)
    index = np.lexsort((temp[:, 1], temp[:, 0]))
    #print("index",index)
    #Population.printPop()
    Population = Population.sort_index(index)

    return Population

def EnvironmentalSelection2(Population,Popsize):
    miu=Population.length
    n_var=Population.decs.shape[1]
    n_obj=Population.objs.shape[1]
    #利用欧式距离和非支配关系排序
    Population=dist_sort(Population)
    sorted_temp_EXA3=np.concatenate((Population.decs,Population.objs),axis=1)
    
    p=np.zeros(sorted_temp_EXA3.shape[0])
    #计算频率
    sum_value=0.0
    for t in range (0,n_var):
        max_value = np.max(sorted_temp_EXA3[:, t])
        min_value = np.min(sorted_temp_EXA3[:, t])
        diff_value = (max_value-min_value)**2
        sum_value=sum_value+diff_value
    oumiga=np.sqrt(sum_value/(n_var))
    if(oumiga==0.0):
        oumiga=1.0
        #print("分母为0，出错1")
    for t1 in range (sorted_temp_EXA3.shape[0]):
        sum_r=0.0
        for t2 in range (sorted_temp_EXA3.shape[0]):
            temp_v=sorted_temp_EXA3[t1,0:n_var]-sorted_temp_EXA3[t2,0:n_var]
            l2=np.linalg.norm(temp_v, ord=2) #求解L2范数
            u=l2/oumiga
            fai=np.exp(-(u*u/2))/np.sqrt(2*math.pi)
            r1=fai*(miu-t2)/(oumiga*miu)
            sum_r=sum_r+r1
        r2=sum_r/sorted_temp_EXA3.shape[0]
        p[t1]=r2
    sorted_p=np.argsort(p)[::-1]  #降序即：取频率高的
    sorted_temp_EXA4=sorted_temp_EXA3[sorted_p]

    Population=pop_class(sorted_temp_EXA4[:,0:n_var],sorted_temp_EXA4[:,n_var:n_var+n_obj])

    if Population.length > Popsize:
        Population = Population.select_index(0, Popsize)

    CrowdDis = Crowding(Population.decs) 

    return Population,CrowdDis

def EnvironmentalSelection3(Population,Popsize):

    n = Population.length
    dist = distance.pdist(Population.decs)
    dist = distance.squareform(dist)
    
    max_vals = np.max(Population.decs, axis=0)
    min_vals = np.min(Population.decs, axis=0)
    ranges = max_vals - min_vals
    product = np.prod(ranges)
    feature_count = Population.decs.shape[1]
    V = 0.2 * product ** (1. / feature_count)

    DominationX = np.zeros((n,n))

    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] > V:
                continue
            L1 = Population.getObj(i) < Population.getObj(j)
            L2 = Population.getObj(i) > Population.getObj(j)
            if np.all(L1 | (~L2)):
                DominationX[i, j] = 0
                DominationX[j, i] = 1
            elif np.all(L2 | (~L1)):
                DominationX[i, j] = 1
                DominationX[j, i] = 0
    
    LocalC = np.zeros(n)
    for i in range(n):
        tmp = dist[i, :]
        index = tmp < V
        LocalC[i] = np.sum(DominationX[i, index]) / np.sum(index)

    '''
    LocalD = np.zeros(n)
    for i in range(n):
        LocalD[i] = np.sum(DominationX[:, i]) / n
    LocalD.reshape(n,1)
    '''

    sorted_temp_EXA3=np.concatenate((Population.decs,Population.objs),axis=1)
    p=np.zeros(sorted_temp_EXA3.shape[0])
    #计算频率
    sum_value=0.0
    n_var=Population.decs.shape[1]
    n_obj=Population.objs.shape[1]
    for t in range (0,n_var):
        max_value = np.max(sorted_temp_EXA3[:, t])
        min_value = np.min(sorted_temp_EXA3[:, t])
        diff_value = (max_value-min_value)**2
        sum_value=sum_value+diff_value
    oumiga=np.sqrt(sum_value/(n_var))
    if(oumiga==0.0):
        oumiga=1.0

    dist[dist > V] = 0
    u=dist/oumiga
    fai=np.exp((-u*u/2))/np.sqrt(2*math.pi)
    r1=fai/oumiga#*LocalD
    sum_r=np.sum(r1, axis=1, keepdims=True)
    p=sum_r/sorted_temp_EXA3.shape[0]

    CrowdDis=p
    LocalC=LocalC.reshape(1,len(LocalC))
    CrowdDis=CrowdDis.reshape(1,len(CrowdDis))
    temp=np.concatenate((LocalC.T, (-CrowdDis).T), axis=1)
    index = np.lexsort((temp[:, 1], temp[:, 0]))
    Population = Population.sort_index(index)

    if Population.length > Popsize:
        Population = Population.select_index(0, Popsize)

    CrowdDis = Crowding(Population.decs) 

    return Population,CrowdDis