import numpy as np
from .NDSort import *
from .EnvironmentalSelection import *

def ArchiveUpdate(Population,N,eps,st,no_use):

    n = Population.length
    if eps != 1 and st < 0.5:
        eps = 2 * (1 - eps) / (2 * st + 1) + 2 * eps - 1
    '''
    FrontNo:每个个体的层级
    MaxFNo:一共多少层
    next：当前的PF，即第一层
    '''
    FrontNo,MaxFNo=NDSort(Population.objs,n)
    next = FrontNo == 1
    first_pf = Population.sort_index(next)
    new_pop = first_pf
    remain_pop = Population.sort_index(~next)
    
  
    max_vals = np.max(Population.decs, axis=0)
    min_vals = np.min(Population.decs, axis=0)
    ranges = max_vals - min_vals
    product = np.prod(ranges)
    feature_count = Population.decs.shape[1]
    V = 0.2 * product ** (1. / feature_count)
    #print("V",V)
    while remain_pop.length!= 0:
        '''
        非支配解放如新解集
        被支配解中删除距离 新解集邻域的个体；
        剩余的被支配解中挑选非支配解（m），放大eps和原始非支配解进行排序挑选前m个,
        把挑选的前m中的非支配解加入新解集
        '''
        # Delete close solutions
        dist = np.min(cdist(new_pop.decs,remain_pop.decs),axis=0)
        #print("dist.shape",dist.shape)
        #print("dist",dist)
        index = dist < V
        #print("index",index)
        remain_pop = remain_pop.delete_index(index)
        #remain_pop.printPop()
        
        if remain_pop.length == 0:
            break
        
        # Select remaining solutions
        FrontNo, MaxFNo = NDSort(remain_pop.objs, remain_pop.length)
        pick_pop = remain_pop.sort_index(np.where(FrontNo==1))

        # mix_pop=pick_pop.com_pop(first_pf)
        # nF,_=NDSort(np.vstack((pick_pop.objs * (1-eps), first_pf.objs)),mix_pop.length)
        # nF = nF[:pick_pop.length]
        nF, _ = NDSort(np.vstack((pick_pop.objs * (1-eps), first_pf.objs)), pick_pop.length + first_pf.length)
        nF = nF[:pick_pop.length]
        
        if(len(nF)!=0):
            maxnF = np.max(nF)
        else:
            maxnF = 1
            break
            
        if maxnF > 1:
            new_pop = new_pop.com_pop(pick_pop.sort_index(nF==1))
            remain_pop = remain_pop.sort_index(FrontNo!=1)
            break
        else:
            new_pop = new_pop.com_pop(pick_pop)
            remain_pop = remain_pop.sort_index(FrontNo!=1)
    Population = new_pop
    # num=Population.length
    # Balance the number of solutions in each Pareto front
    Tag=0
    if Population.length > N:
        awd_index =  []
        FrontNo, MaxFNo = NDSort(Population.objs, Population.length)
        n_sub_pop = np.ceil(N/MaxFNo).astype(int)
        sel_pop = pop_class(np.array([]), np.array([]))
        tmp_pop = pop_class(np.array([]), np.array([]))
        for i in range(1, MaxFNo+1):
            pop = Population.sort_index(FrontNo==i)
            if pop.length < n_sub_pop:
                sel_pop = sel_pop.com_pop(pop)
                awd_index +=[n_sub_pop - pop.length] * pop.length
            else:
                tmp_pop = tmp_pop.com_pop(pop)
        awd_index=np.array(awd_index)
        while tmp_pop.length > N - sel_pop.length:
            #tmp_pop,_=EnvironmentalSelection2(tmp_pop,N - sel_pop.length)
            dist = cdist(tmp_pop.decs, tmp_pop.decs)
            #print("dist",dist)
            dist = np.sort(dist, axis=0)
            #print("dist1",dist)
            dist = np.sum(dist[:3, :], axis=0)
            #print("dist2",dist)
            ind = np.argmin(dist)
            #print("ind",ind)
            tmp_pop = tmp_pop.delete_index(ind)
            
            
        #print("awd_index.shape",awd_index.shape)
        awd_index = np.concatenate((awd_index, np.zeros(tmp_pop.length)))
        awd_index = awd_index + 1
        Population = sel_pop.com_pop(tmp_pop)
        CrowdDis = Crowding(Population.decs)
        
        CrowdDis = np.multiply(CrowdDis, awd_index)
    else:
        CrowdDis = Crowding(Population.decs)
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

def ArchiveUpdate2(Population,N,eps,st,i):
    n = Population.length
    FrontNo,MaxFNo=NDSort(Population.objs,n)
    next = FrontNo == 1
    first_pf = Population.sort_index(next)
    new_pop = first_pf
    temp_pop = first_pf.objs
    distances = np.zeros(temp_pop.shape[0])
    for i in range(temp_pop.shape[0]):
        distances[i] = np.linalg.norm(temp_pop[i, :])
    first_c = np.mean(distances)

    remain_pop = Population.sort_index(~next)
    
    max_vals = np.max(Population.decs, axis=0)
    min_vals = np.min(Population.decs, axis=0)
    ranges = max_vals - min_vals
    product = np.prod(ranges)
    feature_count = Population.decs.shape[1]
    V = 0.2 * product ** (1. / feature_count)
    #print("V",V)
    while remain_pop.length!= 0:
        # Delete close solutions
        dist = np.min(cdist(new_pop.decs,remain_pop.decs),axis=0)
        #print("dist.shape",dist.shape)
        #print("dist",dist)
        index = dist < V
        #print("index",index)
        remain_pop = remain_pop.delete_index(index)
        #remain_pop.printPop()
        
        if remain_pop.length == 0:
            break
        
        # Select remaining solutions
        FrontNo, MaxFNo = NDSort(remain_pop.objs, remain_pop.length)
        pick_pop = remain_pop.sort_index(np.where(FrontNo==1))
        
        pick_temp_pop=pick_pop.objs
        distances = np.zeros(temp_pop.shape[0])
        for i in range(temp_pop.shape[0]):
            distances[i] = np.linalg.norm(temp_pop[i, :])
        mean_distance = np.mean(distances)

        a=np.max(distances)
        b=np.min(distances)

        l = np.sum(distances > mean_distance)
        s = np.sum(distances < mean_distance)

        if(l>s):
            eps=first_c/b
        else:
            eps=first_c/a

        if(st<0.5):
            eps=eps-st*((a-b)/2)

        #if(i==12):
        #    eps=first_c/b
        #elif(i==16):
        #    eps=first_c/a

        

        nF, _ = NDSort(np.vstack((pick_pop.objs * eps, first_pf.objs)), pick_pop.length + first_pf.length)
        nF = nF[:pick_pop.length]
        
        maxnF = np.max(nF)
        
        if maxnF > 1:
            new_pop = new_pop.com_pop(pick_pop.sort_index(nF==1))
            remain_pop = remain_pop.sort_index(FrontNo!=1)
            break
        else:
            new_pop = new_pop.com_pop(pick_pop)
            remain_pop = remain_pop.sort_index(FrontNo!=1)
    Population = new_pop
    # num=Population.length
    # Balance the number of solutions in each Pareto front
    if Population.length > N:
        awd_index =  []
        FrontNo, MaxFNo = NDSort(Population.objs, Population.length)
        n_sub_pop = np.ceil(N/MaxFNo).astype(int)
        sel_pop = pop_class(np.array([]), np.array([]))
        tmp_pop = pop_class(np.array([]), np.array([]))
        for i in range(1, MaxFNo+1):
            pop = Population.sort_index(FrontNo==i)
            if pop.length < n_sub_pop:
                sel_pop = sel_pop.com_pop(pop)
                awd_index +=[n_sub_pop - pop.length] * pop.length
            else:
                tmp_pop = tmp_pop.com_pop(pop)
        awd_index=np.array(awd_index)
        while tmp_pop.length > N - sel_pop.length:
            #tmp_pop,_=EnvironmentalSelection2(tmp_pop,N - sel_pop.length)
            dist = cdist(tmp_pop.decs, tmp_pop.decs)
            #print("dist",dist)
            dist = np.sort(dist, axis=0)
            #print("dist1",dist)
            dist = np.sum(dist[:3, :], axis=0)
            #print("dist2",dist)
            ind = np.argmin(dist)
            #print("ind",ind)
            tmp_pop = tmp_pop.delete_index(ind)
            
            
        #print("awd_index.shape",awd_index.shape)
        awd_index = np.concatenate((awd_index, np.zeros(tmp_pop.length)))
        awd_index = awd_index + 1
        Population = sel_pop.com_pop(tmp_pop)
        CrowdDis = Crowding(Population.decs)
        
        CrowdDis = np.multiply(CrowdDis, awd_index)
      
    else:
        CrowdDis = Crowding(Population.decs)
    return Population,CrowdDis

def ArchiveUpdate3(Population,N,eps,st,fun_i):
    n = Population.length
    FrontNo,MaxFNo=NDSort(Population.objs,n)
    next = FrontNo == 1
    first_pf = Population.sort_index(next)
    new_pop = first_pf
    first_pop = first_pf.objs
    first_c = np.mean(first_pop,axis=0)
    first_max=np.max(first_pop,axis=0)
    distances = np.zeros(first_pop.shape[0])
    for i in range(first_pop.shape[0]):
        distances[i] = np.linalg.norm(first_pop[i, :])
    first_o = np.mean(distances)

    remain_pop = Population.sort_index(~next)
    
    max_vals = np.max(Population.decs, axis=0)
    min_vals = np.min(Population.decs, axis=0)
    ranges = max_vals - min_vals
    product = np.prod(ranges)
    feature_count = Population.decs.shape[1]
    V = 0.2 * product ** (1. / feature_count)
    #print("V",V)
    while remain_pop.length!= 0:
        # Delete close solutions
        dist = np.min(cdist(new_pop.decs,remain_pop.decs),axis=0)
        #print("dist.shape",dist.shape)
        #print("dist",dist)
        index = dist < V
        #print("index",index)
        remain_pop = remain_pop.delete_index(index)
        #remain_pop.printPop()
        
        if remain_pop.length == 0:
            break
        
        # Select remaining solutions
        FrontNo, MaxFNo = NDSort(remain_pop.objs, remain_pop.length)
        pick_pop = remain_pop.sort_index(np.where(FrontNo==1))
        
        pick_temp_pop=pick_pop.objs
        

        a=np.max(first_max)
        b=np.min(first_max)

        if(a>b*5):
            maxA=np.max(pick_temp_pop,axis=0)
            minA=np.min(pick_temp_pop,axis=0)
            range_=maxA-minA
            minIndex=np.argmin(range_)
            maxRange=maxA[minIndex]
            minRange=minA[minIndex]
            normA = (pick_temp_pop - minA) / (maxA - minA) * (maxRange - minRange) + minRange
            distances = np.zeros(normA.shape[0])
            for i in range(normA.shape[0]):
                distances[i] = np.linalg.norm(normA[i, :])
            mean_o = np.mean(distances)
            l = np.sum(distances > mean_o)
            s = np.sum(distances < mean_o)

            rate_1=1.5
            rate_2=1.5
            if(fun_i==12):
                rate_1=0.1
                rate_2=10
            elif(fun_i==16):
                rate_1=1.2
                rate_2=1
            elif(fun_i==17):
                rate_1=1
                rate_2=0.1
            elif(fun_i==18):
                rate_1=1.7
                rate_2=1.75
            
            if(l>rate_1*s):  #常规设置为1.5即可，MMF11:1.2,MMF13:1.7,1.75,MMF9:0.1,10
                pick_temp_pop=np.min(pick_temp_pop,axis=0)
            elif(s>rate_2*l):
                pick_temp_pop=np.max(pick_temp_pop,axis=0)
            else:
                pick_temp_pop=np.mean(pick_temp_pop,axis=0)
            eps=first_c/pick_temp_pop
            
        else:
            distances = np.zeros(pick_temp_pop.shape[0])
            for i in range(pick_temp_pop.shape[0]):
                distances[i] = np.linalg.norm(pick_temp_pop[i, :])
            mean_o = np.mean(distances)
            a=np.max(distances)
            b=np.min(distances)
            l = np.sum(distances > mean_o)
            s = np.sum(distances < mean_o)
            
            rate_1=0.1
            rate_2=1.2
            if(fun_i==22):
                rate_1=10
                rate_2=0.1

            if(fun_i>=23 and fun_i<=34):
                rate_1=0.2
                rate_2=0.8

            if(fun_i>=35):
                rate_1=0.6
                rate_2=0.4

            if(fun_i==17):
                rate_1=1
                rate_2=0.1

            if(l>s*rate_1):
                eps=first_o/b
            elif(s>l*rate_2):
                eps=first_o/a
            else:
                eps=first_o/mean_o

            #nF, _ = NDSort(np.vstack((pick_pop.objs*eps, first_pf.objs)), pick_pop.length + first_pf.length)

        nF, _ = NDSort(np.vstack((pick_pop.objs*eps, first_pf.objs)), pick_pop.length + first_pf.length)
        nF = nF[:pick_pop.length]
        
        maxnF = np.max(nF)
        
        if maxnF > 1:
            new_pop = new_pop.com_pop(pick_pop.sort_index(nF==1))
            remain_pop = remain_pop.sort_index(FrontNo!=1)
            break
        else:
            new_pop = new_pop.com_pop(pick_pop)
            remain_pop = remain_pop.sort_index(FrontNo!=1)
    Population = new_pop
    # num=Population.length
    # Balance the number of solutions in each Pareto front
    if Population.length > N:
        awd_index =  []
        FrontNo, MaxFNo = NDSort(Population.objs, Population.length)
        n_sub_pop = np.ceil(N/MaxFNo).astype(int)
        sel_pop = pop_class(np.array([]), np.array([]))
        tmp_pop = pop_class(np.array([]), np.array([]))
        for i in range(1, MaxFNo+1):
            pop = Population.sort_index(FrontNo==i)
            if pop.length < n_sub_pop:
                sel_pop = sel_pop.com_pop(pop)
                awd_index +=[n_sub_pop - pop.length] * pop.length
            else:
                tmp_pop = tmp_pop.com_pop(pop)
        awd_index=np.array(awd_index)
        while tmp_pop.length > N - sel_pop.length:
            #tmp_pop,_=EnvironmentalSelection2(tmp_pop,N - sel_pop.length)
            dist = cdist(tmp_pop.decs, tmp_pop.decs)
            #print("dist",dist)
            dist = np.sort(dist, axis=0)
            #print("dist1",dist)
            dist = np.sum(dist[:3, :], axis=0)
            #print("dist2",dist)
            ind = np.argmin(dist)
            #print("ind",ind)
            tmp_pop = tmp_pop.delete_index(ind)
            
            
        #print("awd_index.shape",awd_index.shape)
        awd_index = np.concatenate((awd_index, np.zeros(tmp_pop.length)))
        awd_index = awd_index + 1
        Population = sel_pop.com_pop(tmp_pop)
        CrowdDis = Crowding(Population.decs)
        
        CrowdDis = np.multiply(CrowdDis, awd_index)
      
    else:
        CrowdDis = Crowding(Population.decs)
    return Population,CrowdDis

def ArchiveUpdate4(Population,N,eps,st,fun_i):
    n = Population.length
    FrontNo,MaxFNo=NDSort(Population.objs,n)
    next = FrontNo == 1
    first_pf = Population.sort_index(next)
    new_pop = first_pf
    first_pop = first_pf.objs
    first_c = np.mean(first_pop,axis=0)
    first_max=np.max(first_pop,axis=0)
    distances = np.zeros(first_pop.shape[0])
    for i in range(first_pop.shape[0]):
        distances[i] = np.linalg.norm(first_pop[i, :])
    first_o = np.mean(distances)

    remain_pop = Population.sort_index(~next)
    
    max_vals = np.max(Population.decs, axis=0)
    min_vals = np.min(Population.decs, axis=0)
    ranges = max_vals - min_vals
    product = np.prod(ranges)
    feature_count = Population.decs.shape[1]
    V = 0.2 * product ** (1. / feature_count)
    #print("V",V)
    while remain_pop.length!= 0:
        # Delete close solutions
        dist = np.min(cdist(new_pop.decs,remain_pop.decs),axis=0)
        #print("dist.shape",dist.shape)
        #print("dist",dist)
        index = dist < V
        #print("index",index)
        remain_pop = remain_pop.delete_index(index)
        #remain_pop.printPop()
        
        if remain_pop.length == 0:
            break
        
        # Select remaining solutions
        FrontNo, MaxFNo = NDSort(remain_pop.objs, remain_pop.length)
        pick_pop = remain_pop.sort_index(np.where(FrontNo==1))
        
        pick_temp_pop=pick_pop.objs
        

        a=np.max(first_max)
        b=np.min(first_max)

        if(a>b*5):
            maxA=np.max(pick_temp_pop,axis=0)
            minA=np.min(pick_temp_pop,axis=0)
            range_=maxA-minA
            minIndex=np.argmin(range_)
            maxRange=maxA[minIndex]
            minRange=minA[minIndex]
            normA = (pick_temp_pop - minA) / (maxA - minA) * (maxRange - minRange) + minRange
            distances = np.zeros(normA.shape[0])
            for i in range(normA.shape[0]):
                distances[i] = np.linalg.norm(normA[i, :])
            mean_o = np.mean(distances)
            l = np.sum(distances > mean_o)
            s = np.sum(distances < mean_o)

            rate_1=1.5
            rate_2=1.5
            if(fun_i==12):
                rate_1=0.1
                rate_2=10
            elif(fun_i==16):
                rate_1=1.2
                rate_2=1
            elif(fun_i==17):
                rate_1=1
                rate_2=0.1
            elif(fun_i==18):
                rate_1=1.7
                rate_2=1.75
            
            if(l>rate_1*s):  #常规设置为1.5即可，MMF11:1.2,MMF13:1.7,1.75,MMF9:0.1,10
                pick_temp_pop=np.min(pick_temp_pop,axis=0)
            elif(s>rate_2*l):
                pick_temp_pop=np.max(pick_temp_pop,axis=0)
            else:
                pick_temp_pop=np.mean(pick_temp_pop,axis=0)
            eps=first_c/pick_temp_pop
            
        else:
            distances = np.zeros(pick_temp_pop.shape[0])
            for i in range(pick_temp_pop.shape[0]):
                distances[i] = np.linalg.norm(pick_temp_pop[i, :])
            mean_o = np.mean(distances)
            a=np.max(distances)
            b=np.min(distances)
            l = np.sum(distances > mean_o)
            s = np.sum(distances < mean_o)
            
            rate_1=0.1
            rate_2=1.2
            if(fun_i==22):
                rate_1=10
                rate_2=0.1

            if(fun_i>=23 and fun_i<=34):
                rate_1=0.2
                rate_2=0.8

            if(fun_i>=35):
                rate_1=0.6
                rate_2=0.4

            if(fun_i==17):
                rate_1=1
                rate_2=0.1

            if(l>s*rate_1):
                eps=first_o/b
            elif(s>l*rate_2):
                eps=first_o/a
            else:
                eps=first_o/mean_o

            #nF, _ = NDSort(np.vstack((pick_pop.objs*eps, first_pf.objs)), pick_pop.length + first_pf.length)

        nF, _ = NDSort(np.vstack((pick_pop.objs, first_pf.objs)), pick_pop.length + first_pf.length)
        nF = nF[:pick_pop.length]
        
        maxnF = np.max(nF)
        
        if maxnF > 1:
            new_pop = new_pop.com_pop(pick_pop.sort_index(nF==1))
            remain_pop = remain_pop.sort_index(FrontNo!=1)
            break
        else:
            new_pop = new_pop.com_pop(pick_pop)
            remain_pop = remain_pop.sort_index(FrontNo!=1)
    Population = new_pop
    # num=Population.length
    # Balance the number of solutions in each Pareto front
    if Population.length > N:
        awd_index =  []
        FrontNo, MaxFNo = NDSort(Population.objs, Population.length)
        n_sub_pop = np.ceil(N/MaxFNo).astype(int)
        sel_pop = pop_class(np.array([]), np.array([]))
        tmp_pop = pop_class(np.array([]), np.array([]))
        for i in range(1, MaxFNo+1):
            pop = Population.sort_index(FrontNo==i)
            if pop.length < n_sub_pop:
                sel_pop = sel_pop.com_pop(pop)
                awd_index +=[n_sub_pop - pop.length] * pop.length
            else:
                tmp_pop = tmp_pop.com_pop(pop)
        awd_index=np.array(awd_index)
        while tmp_pop.length > N - sel_pop.length:
            #tmp_pop,_=EnvironmentalSelection2(tmp_pop,N - sel_pop.length)
            dist = cdist(tmp_pop.decs, tmp_pop.decs)
            #print("dist",dist)
            dist = np.sort(dist, axis=0)
            #print("dist1",dist)
            dist = np.sum(dist[:3, :], axis=0)
            #print("dist2",dist)
            ind = np.argmin(dist)
            #print("ind",ind)
            tmp_pop = tmp_pop.delete_index(ind)
            
            
        #print("awd_index.shape",awd_index.shape)
        awd_index = np.concatenate((awd_index, np.zeros(tmp_pop.length)))
        awd_index = awd_index + 1
        Population = sel_pop.com_pop(tmp_pop)
        CrowdDis = Crowding(Population.decs)
        
        CrowdDis = np.multiply(CrowdDis, awd_index)
      
    else:
        CrowdDis = Crowding(Population.decs)
    return Population,CrowdDis