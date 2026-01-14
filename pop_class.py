import numpy as np
import torch

class pop_class:
    def __init__(self,decs,objs):
        self.decs=decs
        self.objs=objs
        self.length=decs.shape[0]
        if(self.length!=0):
            self.n_var=decs.shape[1]
            self.n_obj=objs.shape[1]
            self.pop=np.concatenate((decs,objs),axis=1)
            #self.feature_pop=torch.from_numpy(self.pop).contiguous().view(-1) 
    def getDec(self,i):
        return self.decs[i,:]
    def getObj(self,i):
        return self.objs[i,:]
    def sort_index(self,index):
        p=pop_class(self.decs[index],self.objs[index])
        return p
    def select_index(self,l,u):  
        p=pop_class(self.decs[l:u],self.objs[l:u])
        return p
    def delete_index(self,index):
        p=pop_class(np.delete(self.decs, index, axis=0),np.delete(self.objs, index, axis=0))
        return p
    def com_pop(self,p):
        if(self.length==0):
            c_decs= np.copy(p.decs)
            c_objs= np.copy(p.objs)
        else:
            c_decs=np.vstack((self.decs,p.decs))
            c_objs=np.vstack((self.objs,p.objs))
        c=pop_class(c_decs,c_objs)
        return c
    def printPop(self):
        print("decs:")
        for i in range (self.length):
            print("i",i)
            print("decs[i]",self.decs[i,:])
        '''
        print("objs:")
        for i in range (self.length):
            print("objs[i]",self.objs[i,:])
        '''

#pop_class测试代码
'''
decs=np.array([[1,2],[5,4],[2,4],[6,5],[7,8],[1,0]])
objs=np.array([[1,1],[5,5],[2,2],[6,6],[7,7],[1,1]])
Population=pop_class(decs,objs)

print("Population.getDec(2):",Population.getDec(2))
print("Population.getObj(2):",Population.getObj(2))

index=[True,True,True,False,False,False]
a=Population.sort_index(index)
print("sorted:bool")
a.printPop()
b=Population.delete_index(index)
print("sorted:bool")
b.printPop()
'''
'''
index=np.array([5,4,3,2,1,0])
b=Population.sort_index(index)  
print("sorted:int")
print(b.printPop())
c=Population.select_index(2,3+1)  
print("select_index:幅度")
c.printPop()
index=[0,2,4]
d=Population.delete_index(index)  
print("delete_index")
d.printPop()
q=pop_class(np.array([[1.2,1.2],[5.6,5.6],[7.8,7.8]]),np.array([[1.2,1.2],[5.6,5.6],[7.8,7.8]]))
e=Population.com_pop(q)
print("comb_pop")
e.printPop()
'''
