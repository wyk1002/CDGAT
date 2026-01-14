import numpy as np
import mat73

def NDSort(PopObj,n):
    N, M = PopObj.shape
    nSort = N

    if M < 3 or N < 500:
        # Use efficient non-dominated sort with sequential search (ENS-SS)
        FrontNo, MaxFNo = ENS_SS(PopObj, nSort)
    else:
        # Use tree-based efficient non-dominated sort (T-ENS)
        FrontNo, MaxFNo = T_ENS(PopObj, nSort)
    return FrontNo, MaxFNo

def unique_rows_indices(arr):
    _, indices, inverse_indices = np.unique(arr, axis=0, return_index=True, return_inverse=True)
    return indices,inverse_indices

def ENS_SS(PopObj, nSort):
    #按行去重复并且返回去重之后的索引数组
    indices,Loc= unique_rows_indices(PopObj)
    PopObj=PopObj[indices]
    #print("Loc.shape",Loc.shape)
    #print("Loc",Loc)
    #print("PopObj",PopObj)
    Table,_ = np.histogram(Loc, bins=np.arange(0,np.max(Loc) + 2))
    '''
    print("Table.shape",Table.shape)
    print("Table",Table)
    y=0
    l=0
    for i in range(len(Table)):
        if(Table[i]==1):
            y=y+1
        elif(Table[i]==2):
            l=l+1
            print(i)
    print("y",y)
    print("l",l)
    '''
    N, M = PopObj.shape
    FrontNo = np.full(N, np.inf)
    MaxFNo = 0

    while np.sum(Table[FrontNo < np.inf]) < min(nSort, len(Loc)):
        MaxFNo += 1
        for i in range(N):
            if FrontNo[i] == np.inf:
                Dominated = False
                for j in range(i - 1, -1, -1):
                    if FrontNo[j] == MaxFNo:
                        m = 1
                        while m < M and PopObj[i, m] >= PopObj[j, m]:
                            m += 1
                        Dominated = m == M
                        if Dominated or M == 2:
                            break
                if not Dominated:
                    FrontNo[i] = MaxFNo

    FrontNo = FrontNo[Loc]
    return FrontNo, MaxFNo

def T_ENS(PopObj, nSort):
    indices,Loc= unique_rows_indices(PopObj)
    PopObj=PopObj[indices]
    #print("Loc.shape",Loc.shape)
    #print("Loc",Loc)
    #print("PopObj",PopObj)
    Table,_ = np.histogram(Loc, bins=np.arange(0,np.max(Loc) + 2))
    N, M = PopObj.shape
    FrontNo = np.full(N, np.inf)
    MaxFNo = 0
    Forest = np.zeros(N, dtype=int)
    Children = np.full((N, M - 1), -1,dtype=int)
    LeftChild = np.full(N, M, dtype=int)
    Father = np.full(N, -1,dtype=int)
    Brother = np.full(N, M, dtype=int)
    ORank = ORank = np.argsort(PopObj[:,1:M], axis=1)[..., ::-1] + 1
    #print("ORank",ORank)
    while np.sum(Table[FrontNo < np.inf]) < min(nSort, len(Loc)):
        MaxFNo += 1
        if(MaxFNo>=N):
            break
        root = np.where(FrontNo == np.inf)[0][0]
        Forest[MaxFNo] = root
        FrontNo[root] = MaxFNo
        #print("MaxFNo",MaxFNo)
        #print("root",root)
        for p in range(N):
            if FrontNo[p] == np.inf:
                Pruning = np.zeros(N, dtype=int)
                q = Forest[MaxFNo]
                '''
                if(p==178):
                    print("q",q)
                '''
                while True:
                    m = 0
                    '''
                    if(p==178 and q==39):
                        print("q",q)
                        print("ORank[q, m]",ORank[q, m])
                        print("PopObj[p, ORank[q, m]]",PopObj[p, ORank[q, m]])
                        '''
                    while m < M - 1 and PopObj[p, ORank[q, m]] >= PopObj[q, ORank[q, m]]:
                        m += 1
                        '''
                    if(p==178 and q==39):
                        print("m",m)
                        '''
                    if m == M - 1:
                        break
                    else:
                        Pruning[q] = m
                        
                        if LeftChild[q] <= Pruning[q]:
                            q = Children[q, LeftChild[q]]
                        else:
                            while (Father[q]!=-1) and Brother[q] > Pruning[Father[q]]:
                                q = Father[q]
                            '''
                            if(p==178):
                                print("q2",q)
                                print("Father[q2]",Father[q])
                                print("Father:",Father)
                            '''
                            if (Father[q]!=-1):
                                q = Children[Father[q], Brother[q]]
                            else:
                                break
                
                if m < M - 1:
                    FrontNo[p] = MaxFNo
                    q = Forest[MaxFNo]
                    
                    while (Children[q, Pruning[q]]!=-1):
                        q = Children[q, Pruning[q]]
                    
                    Children[q, Pruning[q]] = p
                    Father[p] = q
                    
                    if LeftChild[q] > Pruning[q]:
                        Brother[p] = LeftChild[q]
                        LeftChild[q] = Pruning[q]
                    else:
                        bro = Children[q, LeftChild[q]]
                        
                        while Brother[bro] < Pruning[q]:
                            bro = Children[q, Brother[bro]]
                        
                        Brother[p] = Brother[bro]
                        Brother[bro] = Pruning[q]
    
    FrontNo = FrontNo[Loc]
    return FrontNo, MaxFNo