import numpy as np
import math
import random

class paretoF:
    def __init__(self):
        self.f = []   #存储索引

    def add_item(self, item):
        self.f.append(item)

    def remove_item(self, item):
        if item in self.f:
            self.f.remove(item)
        else:
            print("元素未找到！")

    def clear_list(self):
        self.f.clear()

    def get_length(self):
        return len(self.f)

    def isEmpty(self):
        if(len(self.f)==0):
            return 0
        else:
            return 1    

    
class single_individual:
    def __init__(self):
        self.n = 0
        self.p = []  #存储支配个体的索引
    
    def add_item(self, item):
        self.p.append(item)

    def get_length(self):
        return len(self.p)

def non_domination_scd_sort(x, n_obj, n_var):
    N_particle = x.shape[0]
    empty_arr = np.zeros((N_particle, 1))#np.full((N_particle, 1), -1.0)#np.zeros((N_particle, 1))
    #print("x.shape",x.shape)
    #print("empty_arr.shape",empty_arr.shape)
    x = np.concatenate((x, empty_arr), axis=1)
    front = 0
    F = []            #存储各个前沿中的个体的索引
    individual = []
    for i in range(N_particle):  #初始化个体数组
        #print("i1",i)
        individual.append(single_individual())
        #print("i2",i)
        F.append(paretoF())

    F.append(paretoF())
    
    for i in range(N_particle):
        # individual[i].n Number of individuals that dominate this individual支配当前个体的个体数量 
        # individual(i).p = [] Individuals which this individual dominate当前个体支配的个体
        for j in range(N_particle):
            dom_less = 0
            dom_equal = 0
            dom_more = 0
            for k in range (n_obj):
                if (x[i,n_var + k] < x[j,n_var + k]):
                    dom_less = dom_less + 1
                elif (x[i,n_var + k] == x[j,n_var + k]):  
                    dom_equal = dom_equal + 1
                else:
                    dom_more = dom_more + 1
                
            if (dom_less == 0 and dom_equal != n_obj):#!=特殊字符表示不等于
                individual[i].n = individual[i].n + 1
            elif (dom_more == 0 and dom_equal != n_obj):
                individual[i].add_item(j)   

        if (individual[i].n == 0):
            x[i,n_obj + n_var ] = 0   #帕累托前沿从0开始
            F[front].add_item(i) 

    #print("x",x)
    #for i in range(N_particle):
    #    print("i:",i)
    #    print("individual[i].n",individual[i].n)
    #    print("individual[i].p",individual[i].p)

    while (F[front].get_length()!=0):
        Q = []
        for i in range(F[front].get_length()):  
            if (individual[F[front].f[i]].get_length()!=0):
                for j in range (individual[F[front].f[i]].get_length()):
                    
                    individual[individual[F[front].f[i]].p[j]].n = individual[individual[F[front].f[i]].p[j]].n - 1
                        
                    if (individual[individual[F[front].f[i]].p[j]].n == 0):

                        x[individual[F[front].f[i]].p[j],n_obj + n_var ] = front + 1
                            
                        Q.append(individual[F[front].f[i]].p[j])
                    
        front =  front + 1
        #print("front",front)
        F[front].f = Q

    
    #i=0
    #while (F[i].get_length()!=0): 
    #    print("F[front].f:",F[i].f)
    #    i=i+1

    index_of_fronts=np.argsort(x[:,n_obj + n_var])
    
    sorted_based_on_front=x[np.argsort(x[:,n_obj + n_var ])]

    #print("index_of_fronts",index_of_fronts)
    #print("sorted_based_on_front",sorted_based_on_front)
    
    current_index = 0
    z=np.zeros((N_particle,n_obj + n_var + 4))#np.full((N_particle,n_obj + n_var + 4),-1.0)#np.zeros((N_particle,n_obj + n_var + 4))
    Max_s_front=0
    for k in range (len(F)):
        if(F[k].get_length()!=0):
            #print("k",k)
            #print("F[k].get_length",F[k].get_length())
            Max_s_front=Max_s_front+1
    #print("Max_s_front",Max_s_front)
    #for k in range (sorted_based_on_front.shape[0]):
    sorted_based_on_front[:,n_obj + n_var]=sorted_based_on_front[:,n_obj + n_var]+1
    #print("sorted_based_on_front",sorted_based_on_front)
    for front in range (Max_s_front):
        y = np.zeros((F[front].get_length(), n_obj+n_var+1))
        previous_index = current_index 
        y[0:F[front].get_length(),0:sorted_based_on_front.shape[1]]=sorted_based_on_front[current_index:current_index+F[front].get_length(),:]
        current_index = current_index + F[front].get_length() 
        #print("y.shape",y.shape)
        #print("y1",y)
        #print("front",front)
        #print("F[front].get_length()",F[front].get_length())
        #print("y.shape",y.shape)
        #print("y",y)
        for i in range(n_obj+n_var):
            index_of_objectives=np.argsort(y[:,i])
            #print("index_of_objectives",index_of_objectives)
            #sorted_based_on_objective = [] python使用numpy类型数据无需声明
            #for j in range (index_of_objectives.shape[0])
            #    sorted_based_on_objective(j,:) = y(index_of_objectives(j),:)
            sorted_based_on_objective=y[np.argsort(y[:,i])]
            #print("y.shape",y.shape)
            f_max = sorted_based_on_objective[index_of_objectives.shape[0]-1, i]
            f_min = sorted_based_on_objective[0,  i]
            #print("i",i)
            #print("sorted_based_on_objective",sorted_based_on_objective)
            #print("f_max",f_max)
            #print("f_min",f_min)
            max_length=index_of_objectives.shape[0]-1
            '''
            if(front==0 and i==n_var):
                print(sorted_based_on_objective)
                print("index_of_objectives",index_of_objectives)
                print("f_max",f_max)
                print("f_min",f_min)
            '''
            
            while(n_obj + n_var+1 + i>=y.shape[1]):
                    empty_arr = np.zeros((y.shape[0], 1))
                    y = np.concatenate((y, empty_arr), axis=1)

            if (index_of_objectives.shape[0]==1):
                y[index_of_objectives[0],n_obj + n_var+1  + i] = 1 #If there is only one point in current front
            elif(i>=n_var): 
                y[index_of_objectives[0], n_obj + n_var +1 + i] = 1
                y[index_of_objectives[max_length], n_obj + n_var +1 + i] = 0
            else:
                #print("max_length",max_length)
                #print("index_of_objectives[max_length]",index_of_objectives[max_length]) 
                #print("y[index_of_objectives[max_length],n_obj + n_var+1  + i]",y[index_of_objectives[max_length],n_obj + n_var+1  + i])   
                #print("(f_max - f_min)",(f_max - f_min))
                #if(f_max - f_min==0):
                    #print("index_of_objectives",index_of_objectives)
                    #print("sorted_based_on_objective",sorted_based_on_objective)
                    #print("f_max",f_max)
                    #print("f_min",f_min)
                y[index_of_objectives[max_length],n_obj + n_var+1  + i]= 2*(sorted_based_on_objective[max_length, i]-sorted_based_on_objective[max_length-1, i])/((f_max - f_min))
                y[index_of_objectives[0],n_obj + n_var +1 + i]=2*(sorted_based_on_objective[1, i]-sorted_based_on_objective[0, i])/((f_max - f_min))
            
            #print("i",i)
            #print("y:",y)

            for j in range(1,index_of_objectives.shape[0]-1): #2 : length(index_of_objectives) - 1
                next_obj  = sorted_based_on_objective[j + 1, i]
                previous_obj  = sorted_based_on_objective[j - 1,i]
                if (f_max - f_min == 0): #only one point in the current Front
                    
                    y[index_of_objectives[j],n_obj + n_var +1 + i] = 1
                else:
                    y[index_of_objectives[j],n_obj + n_var  +1+ i] = (next_obj - previous_obj)/(f_max - f_min)
        

        #print("y.shape",y.shape)
        
        #print("y2",y)
        #print("y",y)
        #决策空间距离 
        crowd_dist_var=np.zeros((F[front].get_length(),1))
        for i in range(n_var): 
            crowd_dist_var[:,0] = crowd_dist_var[:,0] + y[:,n_obj + n_var +1+ i]
        
        
        crowd_dist_var = crowd_dist_var / n_var
        avg_crowd_dist_var=np.mean(crowd_dist_var)
        #print("crowd_dist_var",crowd_dist_var)
        #print("avg_crowd_dist_var",avg_crowd_dist_var)
        #目标空间距离
        crowd_dist_obj=np.zeros((F[front].get_length(),1))
        for i in range(n_obj): 
            crowd_dist_obj[:,0] = crowd_dist_obj[:,0] + y[:,n_obj + n_var  + n_var +1+ i]
        
        
        crowd_dist_obj = crowd_dist_obj / n_obj
        avg_crowd_dist_obj=np.mean(crowd_dist_obj)
        #print("crowd_dist_obj",crowd_dist_obj)
        #print("avg_crowd_dist_obj",avg_crowd_dist_obj)

        #特殊拥挤距离
        special_crowd_dist=np.zeros((F[front].get_length(),1))
        for i in range(len(F[front].f)):
            if crowd_dist_obj[i] > avg_crowd_dist_obj or crowd_dist_var[i] > avg_crowd_dist_var:
                special_crowd_dist[i] = max(crowd_dist_obj[i], crowd_dist_var[i])  
            else:
                special_crowd_dist[i] = min(crowd_dist_obj[i], crowd_dist_var[i])

        #print("special_crowd_dist",special_crowd_dist)
        #print("y.shape",y.shape)
        #print("y[0:y.shape[0], n_obj + n_var + 1].shape",y[0:y.shape[0], n_obj + n_var + 1].shape)
        #print("y1",y)
        y[0:y.shape[0], n_obj + n_var + 1] = special_crowd_dist[:,0]
        #print("y2",y)
        y[0:y.shape[0], n_obj + n_var + 2] = crowd_dist_var[:,0]
        #print("y3",y)
        y[0:y.shape[0], n_obj + n_var + 3] = crowd_dist_obj[:,0]

        #print("y3")
        #for m in range(y.shape[0]):
        #    print(y[m])
        #print("y.shape",y.shape)
        #print("crowd_dist_obj.shape",crowd_dist_obj.shape)
        #_, index_sorted_based_crowddist = np.sort(crowd_dist_obj)[::-1].sort()  # sort the particles in the same front according to SCD
        index_sorted_based_crowddist=np.argsort(special_crowd_dist, axis=0)[::-1]
        #print("index_sorted_based_crowddist.shape",index_sorted_based_crowddist.shape)
        y=y[index_sorted_based_crowddist[:,0],:]
        #print("y.shape",y.shape)
        y = y[:, :n_obj + n_var + 4]
        #print("y.shape",y.shape)
        #print("z[previous_index:current_index, :].shape",z[previous_index:current_index, :].shape)
        
        z[previous_index:current_index, :] = y

    sorted_data = z.copy()
    return sorted_data

