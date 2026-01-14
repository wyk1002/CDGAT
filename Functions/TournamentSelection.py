import numpy as np
import mat73
def unique_rows_indices(arr):
    _, indices, inverse_indices = np.unique(arr, axis=0, return_index=True, return_inverse=True)
    return indices,inverse_indices
def TournamentSelection(K,N,varargin):
    varargin=varargin.reshape(len(varargin),1)
    indices,Loc= unique_rows_indices(varargin)
    Fit=varargin[indices]
    temp_rank = np.lexsort(Fit.T)
    rank = np.argsort(temp_rank)
    '''
    print("rank.shape",rank.shape)
    print("rank",rank)
    print("Loc.shape",Loc.shape)
    print("Loc",Loc)
    print("Fit.shape",Fit.shape)
    print("Fit",Fit)
    '''
    Loc=Loc.reshape(len(Loc),1)
    rank=rank.reshape(len(rank),1)
    #print("rank.shape",rank.shape)
    #print("Loc.shape",Loc.shape)
    Parents = np.random.randint(0, varargin.shape[0], size=(K, N))
    #Parents=mat73.loadmat('./data/Parents.mat')['Parents']
    #Parents=Parents.astype(int) 
    #Parents=Parents-1
    #print("Parents.shape",Parents.shape)
    #print("type(Parents)",type(Parents))
    #print("Loc.shape",Loc.shape)
    a=Loc[Parents]
    b=rank[a]
    b=np.squeeze(b)
    #print("b.shape",b.shape)
    best = np.argmin(b, axis=0)
    #print("best.shape",best.shape)
    #print("best",best)
    index=np.zeros((1,N))
    for i in range (N):
        index[0,i]=Parents[best[i],i]
    
    return index
