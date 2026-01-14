from Functions.function import *

def Init(Popsize,n,M,fname,xl,xu):#xl是下界，xu是上界
    #print("xl",xl)
    #print("xu",xu)
    chrom = np.zeros((Popsize, n))
    obj = np.zeros((Popsize, M))
    temp=np.zeros((Popsize, n))
    #Popsize=10

    for i in range(Popsize):
        rand = np.random.rand(n) #* 0.9999 + 0.0001
        chrom[i] = xl + (xu - xl) * rand
        temp[i]=np.copy(chrom[i])
        #print(xu-xl)
        #print((xu - xl) * rand)
        #print(chrom[i])
        obj[i] = eval(fname)(temp[i])
        #print(obj[i])

    
    #for i in range(Popsize):
    #    print("temp[i]",temp[i])
    #    print("chrom[i]",chrom[i])
    #    print("obj[i]",obj[i])

    #while(Popsize==10):
    #    a=1+1
    return chrom,obj
    