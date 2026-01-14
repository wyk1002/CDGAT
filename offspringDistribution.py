from Init import *
import scipy.io as sio
from Functions.Indicator_calculation import*
from Functions.ArchiveUpdate import *
from Functions.TournamentSelection import *
from OperatorGA import *
from LSTM_Net import *


runs=1 #运行次数
N_function=13
Popsize=800 #  用户自定义正整数,根据该正整数及问题维度可确定种群数目N
eps=0.3
p = 0.5
maxFEx=4800

batch_size = 2
epoch=300
generate_epoch=100
pi_value = np.pi
lr=0.001

class Individual:
    def __init__(self):
        self.n = 0
        self.weight=[]
        self.chrom=[]
        self.obj=[]
        self.tag=0
class Individual_1:
    def __init__(self):
        self.weight=[]
        self.chrom=[]
        self.obj=[]
        self.PSS=float('inf')
        self.selecte=0
        self.tag=0
class population:
    def __init__(self, n_var, n_obj, pop):  #n表示当代种群中的个体数量,pop表示当前种群
        self.n_var=n_var
        self.n_obj=n_obj
        self.pop=pop
        self.feature_pop=pop.contiguous().view(-1) 
        

def fixnew(newpoint, min_range, max_range):
    #print("newpoint1",newpoint)
    newpoint = np.maximum(newpoint, min_range)
    newpoint = np.minimum(newpoint, max_range)
    #print("newpoint2",newpoint)
    return newpoint

for i in range(13, N_function+1):
    print("函数：",i)
    hyp=[]
    IGDx=[]
    IGDf=[]
    CR=[]
    PSP=[]
    for j in range(1,runs+1):
        print("运行次数：",j)
        #CEC2019数据集
        if (i==1):
            fname='MMF1'
            n_var=2
            n_obj=2
            xl=np.array([1,-1])     
            xu=np.array([3,1])       
            repoint=np.array([[2,2]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF1truePSPF.mat'
            PS_number=2
        elif (i==2):
            fname='MMF2'
            n_var=2
            n_obj=2
            xl=np.array([0,0])     
            xu=np.array([1,2])       
            repoint=np.array([[2,2]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF2truePSPF.mat'
            PS_number=2
        elif (i==3):
            fname='MMF3'
            n_var=2
            n_obj=2
            xl=np.array([0,0])     
            xu=np.array([1,1.5])       
            repoint=np.array([[2,2]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF3truePSPF.mat'
            PS_number=2
        elif (i==4):
            fname='MMF4'
            n_var=2
            n_obj=2
            xl=np.array([-1,0])     
            xu=np.array([1,2])       
            repoint=np.array([[2,2]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF4truePSPF.mat'
            PS_number=4
        elif (i==5):
            fname='MMF5'
            n_var=2
            n_obj=2
            xl=np.array([1,-1])     
            xu=np.array([3,3])       
            repoint=np.array([[2,2]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF5truePSPF.mat'
            PS_number=4
        elif (i==6):
            fname='MMF6'
            n_var=2
            n_obj=2
            xl=np.array([1,-1])     
            xu=np.array([3,2])       
            repoint=np.array([[2,2]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF6truePSPF.mat'
            PS_number=2
        elif (i==7):
            fname='MMF7'
            n_var=2
            n_obj=2
            xl=np.array([1,-1])     
            xu=np.array([3,1])       
            repoint=np.array([[2,2]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF7truePSPF.mat'
            PS_number=2
        elif (i==8):
            fname='MMF8'
            n_var=2
            n_obj=2
            xl=np.array([-pi_value,0])     
            xu=np.array([pi_value,9])       
            repoint=np.array([[2,2]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF8truePSPF.mat'
            PS_number=4
        elif (i==9):
            fname='SYM_PART_simple'
            n_var=2
            n_obj=2
            xl=np.array([-20,-20])     
            xu=np.array([20,20])       
            repoint=np.array([[2,2]])             
            TruePSPF_path='./TruePS_PF/CEC2019/SYM_PART_simple_turePSPF.mat'
            PS_number=9
        elif (i==10):
            fname='SYM_PART_rotated'
            n_var=2
            n_obj=2
            xl=np.array([-20,-20])     
            xu=np.array([20,20])       
            repoint=np.array([[2,2]])             
            TruePSPF_path='./TruePS_PF/CEC2019/SYM_PART_rotatedtruePSPF.mat'
            PS_number=9
        elif (i==11):
            fname='Omni_test'
            n_var=3
            n_obj=2
            xl=np.array([0,0,0])     
            xu=np.array([6,6,6])       
            repoint=np.array([[5,5]])             
            TruePSPF_path='./TruePS_PF/CEC2019/Omni_testtruePSPF.mat'
            PS_number=27
        elif (i==12):
            fname='MMF9'
            n_var=2
            n_obj=2
            xl=np.array([0.1,0.1])     
            xu=np.array([1.1,1.1])       
            repoint=np.array([[1.21,11]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF9_Reference_PSPF_data.mat'
            #PS_number=2
        elif (i==13):
            fname='MMF14'
            n_var=3
            n_obj=3
            xl=np.array([0,0,0])     
            xu=np.array([1,1,1])       
            repoint=np.array([[2.2,2.2,2.2]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF14_Reference_PSPF_data.mat'
            #PS_number=2
        elif (i==14):
            fname='MMF14_a'
            n_var=3
            n_obj=3
            xl=np.array([0,0,0])     
            xu=np.array([1,1,1])       
            repoint=np.array([[2.2,2.2,2.2]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF14_a_Reference_PSPF_data.mat'
        elif (i==15):
            fname='MMF10'
            n_var=2
            n_obj=2
            xl=np.array([0.1,0.1])     
            xu=np.array([1.1,1.1])       
            repoint=np.array([[1.21,13.2]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF10_Reference_PSPF_data.mat'
            #PS_number=2
        elif (i==16):
            fname='MMF11'
            n_var=2
            n_obj=2
            xl=np.array([0.1,0.1])     
            xu=np.array([1.1,1.1])       
            repoint=np.array([[1.21,15.4]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF11_Reference_PSPF_data.mat'
            #PS_number=2
        elif (i==17):
            fname='MMF12'
            n_var=2
            n_obj=2
            xl=np.array([0,0])     
            xu=np.array([1,1])       
            repoint=np.array([[1.54,1.1]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF12_Reference_PSPF_data.mat'
            #PS_number=2
        elif (i==18):
            fname='MMF13'
            n_var=3
            n_obj=2
            xl=np.array([0.1,0.1,0.1])     
            xu=np.array([1.1,1.1,1.1])       
            repoint=np.array([[1.54,15.4]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF13_Reference_PSPF_data.mat'
            #PS_number=2
        elif (i==19):
            fname='MMF15'
            n_var=3
            n_obj=3
            xl=np.array([0,0,0])     
            xu=np.array([1,1,1])       
            repoint=np.array([[2.5,2.5,2.5]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF15_Reference_PSPF_data.mat'
            PS_number=2
        elif (i==20):
            fname='MMF1_z'
            n_var=2
            n_obj=2
            xl=np.array([1,-1])     
            xu=np.array([3,1])       
            repoint=np.array([[1.1,1.1]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF1_z_Reference_PSPF_data.mat'
            #PS_number=2
        elif (i==21):
            fname='MMF1_e'
            n_var=2
            n_obj=2
            xl=np.array([1,-20])     
            xu=np.array([3,20])       
            repoint=np.array([[1.1,1.1]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF1_e_Reference_PSPF_data.mat'
            #PS_number=2
        elif (i==22):
            fname='MMF15_a'
            n_var=3
            n_obj=3
            xl=np.array([0,0,0])     
            xu=np.array([1,1,1])       
            repoint=np.array([[2.5,2.5,2.5]])             
            TruePSPF_path='./TruePS_PF/CEC2019/MMF15_a_Reference_PSPF_data.mat'
        #IDMP数据集
        elif (i==23):
            fname='IDMPM2T1'
            n_var=2
            n_obj=2
            xl=np.array([-1,-1])     
            xu=np.array([1,1])       
            repoint=np.array([[1,1]])             
            TruePSPF_path='./TruePS_PF/IDMP/IDMPM2T1Reference.mat'
        elif (i==24):
            fname='IDMPM2T2'
            n_var=2
            n_obj=2
            xl=np.array([-1,-1])     
            xu=np.array([1,1])       
            repoint=np.array([[1,1]])             
            TruePSPF_path='./TruePS_PF/IDMP/IDMPM2T2Reference.mat'
        elif (i==25):
            fname='IDMPM2T3'
            n_var=2
            n_obj=2
            xl=np.array([-1,-1])     
            xu=np.array([1,1])       
            repoint=np.array([[1,1]])            
            TruePSPF_path='./TruePS_PF/IDMP/IDMPM2T3Reference.mat'
        elif (i==26):
            fname='IDMPM2T4'
            n_var=2
            n_obj=2
            xl=np.array([-1,-1])     
            xu=np.array([1,1])       
            repoint=np.array([[1,1]])            
            TruePSPF_path='./TruePS_PF/IDMP/IDMPM2T4Reference.mat'
        elif (i==27):
            fname='IDMPM3T1'
            n_var=3
            n_obj=3
            xl=np.array([-1,-1,-1])     
            xu=np.array([1,1,1])       
            repoint=np.array([[1,1,1]])             
            TruePSPF_path='./TruePS_PF/IDMP/IDMPM3T1Reference.mat'
        elif (i==28):
            fname='IDMPM3T2'
            n_var=3
            n_obj=3
            xl=np.array([-1,-1,-1])     
            xu=np.array([1,1,1])       
            repoint=np.array([[1,1,1]])               
            TruePSPF_path='./TruePS_PF/IDMP/IDMPM3T2Reference.mat'
        elif (i==29):
            fname='IDMPM3T3'
            n_var=3
            n_obj=3
            xl=np.array([-1,-1,-1])     
            xu=np.array([1,1,1])       
            repoint=np.array([[1,1,1]])              
            TruePSPF_path='./TruePS_PF/IDMP/IDMPM3T3Reference.mat'
        elif (i==30):
            fname='IDMPM3T4'
            n_var=3
            n_obj=3
            xl=np.array([-1,-1,-1])     
            xu=np.array([1,1,1])       
            repoint=np.array([[1,1,1]])               
            TruePSPF_path='./TruePS_PF/IDMP/IDMPM3T4Reference.mat'
        elif (i==31):
            fname='IDMPM4T1'
            n_var=4
            n_obj=4
            xl=np.array([-1,-1,-1,-1])     
            xu=np.array([1,1,1,1])       
            repoint=np.array([[1,1,1,1]])               
            TruePSPF_path='./TruePS_PF/IDMP/IDMPM4T1Reference.mat'
        elif (i==32):
            fname='IDMPM4T2'
            n_var=4
            n_obj=4
            xl=np.array([-1,-1,-1,-1])     
            xu=np.array([1,1,1,1])       
            repoint=np.array([[1,1,1,1]])              
            TruePSPF_path='./TruePS_PF/IDMP/IDMPM4T2Reference.mat'
        elif (i==33):
            fname='IDMPM4T3'
            n_var=4
            n_obj=4
            xl=np.array([-1,-1,-1,-1])     
            xu=np.array([1,1,1,1])       
            repoint=np.array([[1,1,1,1]])               
            TruePSPF_path='./TruePS_PF/IDMP/IDMPM4T3Reference.mat'
        elif (i==34):
            fname='IDMPM4T4'
            n_var=4
            n_obj=4
            xl=np.array([-1,-1,-1,-1])     
            xu=np.array([1,1,1,1])       
            repoint=np.array([[1,1,1,1]])               
            TruePSPF_path='./TruePS_PF/IDMP/IDMPM4T4Reference.mat'
        elif (i==35):
            fname='MMMOP1A'
            n_var=3
            n_obj=2
            xl=np.array([0,0,0])     
            xu=np.array([1,1,1])                     
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP1A_PFPS.mat'
        elif (i==35):
            fname='MMMOP1A'
            n_var=3
            n_obj=2
            xl=np.array([0,0,0])     
            xu=np.array([1,1,1])                     
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP1A_PFPS.mat'
        elif (i==35):
            fname='MMMOP1A'
            n_var=3
            n_obj=2
            xl=np.array([0,0,0])     
            xu=np.array([1,1,1])                     
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP1A_PFPS.mat'
        elif (i==35):
            fname='MMMOP1A'
            n_var=3
            n_obj=2
            xl=np.array([0,0,0])     
            xu=np.array([1,1,1])                     
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP1A_PFPS.mat'
        elif (i==35):
            fname='MMMOP1A'
            n_var=3
            n_obj=2
            xl=np.array([0,0,0])     
            xu=np.array([1,1,1])                     
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP1A_PFPS.mat'
        elif i==36:
            fname='MMMOP1B'
            n_obj=3
            n_var=7
            xl=np.array([0,0,0,0,0,0,0])
            xu=np.array([1,1,1,1,1,1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP1B_PFPS.mat'
        elif i==37:
            fname='MMMOP2A'
            n_obj=2
            n_var=3
            xl=np.array([0,0,0])
            xu=np.array([1,1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP2A_PFPS.mat'
        elif i== 38:
            fname='MMMOP2B'
            n_obj=3
            n_var=7
            xl=np.array([0,0,0,0,0,0,0])
            xu=np.array([1,1,1,1,1,1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP2B_PFPS.mat'
        elif i== 39:
            fname='MMMOP3A'
            n_obj=2
            n_var=2
            xl=np.array([0,0])
            xu=np.array([1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP3A_PFPS.mat'
        elif i== 40:
            fname='MMMOP3B'
            n_obj=3
            n_var=7
            xl=np.array([0,0,0,0,0,0,0])
            xu=np.array([1,1,1,1,1,1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP3B_PFPS.mat'
        elif i== 41:
            fname='MMMOP3C'
            n_obj=2
            n_var=6
            xl=np.array([0,0,0,0,0,0])
            xu=np.array([1,1,1,1,1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP3C_PFPS.mat'
        elif i== 42:
            fname='MMMOP3D'
            n_obj=3
            n_var=7
            xl=np.array([0,0,0,0,0,0,0])
            xu=np.array([1,1,1,1,1,1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP3D_PFPS.mat'
        elif i== 43:
            fname='MMMOP4A'
            n_obj=2
            n_var=2
            xl=np.array([0,0])
            xu=np.array([1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP4A_PFPS.mat'
        elif i== 44:
            fname='MMMOP4B'
            n_obj=3
            n_var=7
            xl=np.array([0,0,0,0,0,0,0])
            xu=np.array([1,1,1,1,1,1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP4B_PFPS.mat'
        elif i== 45:
            fname='MMMOP4C'
            n_obj=2
            n_var=6
            xl=np.array([0,0,0,0,0,0])
            xu=np.array([1,1,1,1,1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP4C_PFPS.mat'
        elif i== 46:
            fname='MMMOP4D'
            n_obj=3
            n_var=7
            xl=np.array([0,0,0,0,0,0,0])
            xu=np.array([1,1,1,1,1,1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP4D_PFPS.mat'
        elif i== 47:
            fname='MMMOP5A'
            n_obj=2
            n_var=2
            xl=np.array([0,0])
            xu=np.array([1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP5A_PFPS.mat'
        elif i== 48:
            fname='MMMOP5B'
            n_obj=3
            n_var=7
            xl=np.array([0,0,0,0,0,0,0])
            xu=np.array([1,1,1,1,1,1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP5B_PFPS.mat'
        elif i== 49:
            fname='MMMOP5C'
            n_obj=2
            n_var=6
            xl=np.array([0,0,0,0,0,0])
            xu=np.array([1,1,1,1,1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP5C_PFPS.mat'
        elif i== 50:
            fname='MMMOP5D'
            n_obj=3
            n_var=7
            xl=np.array([0,0,0,0,0,0,0])
            xu=np.array([1,1,1,1,1,1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP5D_PFPS.mat'
        elif i== 51:
            fname='MMMOP6A'
            n_obj=2
            n_var=2
            xl=np.array([0,0])
            xu=np.array([1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP6A_PFPS.mat'
        elif i == 52:
            fname='MMMOP6B'
            n_obj=3
            n_var=4
            xl=np.array([0,0,0,0])
            xu=np.array([1,1,1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP6B_PFPS.mat'
        elif i== 53:
            fname='MMMOP6C'
            n_obj=2
            n_var=4
            xl=np.array([0,0,0,0])
            xu=np.array([1,1,1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP6C_PFPS.mat'
        elif i== 54:
            fname='MMMOP6D'
            n_obj=3
            n_var=5
            xl=np.array([0,0,0,0,0])
            xu=np.array([1,1,1,1,1])
            TruePSPF_path='./TruePS_PF/MMMOP/MMMOP6D_PFPS.mat'

        chrom,obj=Init(Popsize,n_var,n_obj,fname,xl,xu)
        Population=pop_class(chrom,obj)
        _,CrowdDis1 = EnvironmentalSelection(Population,Popsize)
        Archive,CrowdDis2 = ArchiveUpdate3(Population,Popsize,eps,0,i)

        train_i=1
        w_size=3
        FEx=800
        dim_n=FEx
        dim=dim_n*(n_var+n_obj)
        #fre=6   #控制使用Transformer使用频率的参数
        net=LSTM_Net(batch_size, lr, epoch, n_var, n_obj, fname, w_size, dim)

        history_pop_Population=[]  #transformer的作用是利用历史种群推导出新种群，history_pop用于存储历史种群
        history_pop_Archive=[]
        while (FEx<=maxFEx):
            print("FEx:",FEx)
            print("train_i",train_i)
            if FEx >= maxFEx * 0.5 and np.random.rand() < p:  #注意随机数定义的不同matlab函数中的rand取值范围是(0,1),而python中的是[0,1)
                if(train_i%6>0):
                    MatingPool2 = TournamentSelection(2, round(Popsize), -CrowdDis2)
                    MatingPool2=MatingPool2.reshape(-1)
                    Offspring = OperatorGA(Archive.sort_index(MatingPool2.astype(int)), xl, xu)
                else:
                    #利用history_pop中的历史个体结合Transformer网络向前预测一代
                    print("Transformer训练")
                    net.train(history_pop_Archive)
                    #网络生成
                    result=net.predict_future(history_pop_Archive)
                    result=result.cpu() 
                    #形状变换
                    last_num=result.shape[0]
                    new_EXA=result[last_num-1,:]
                    new_EXA=torch.reshape(new_EXA, (dim_n, n_var+n_obj))  #EXA表示生成的子代群体
                    Offspring=new_EXA.numpy()
                    Offspring=Offspring[:,0:n_var]
                    #边界控制：
                    Offspring=fixnew(Offspring,xl,xu)
                    
            else:
                if(train_i%6>0):
                    MatingPool1 = TournamentSelection(2, round(Popsize), -CrowdDis1)
                    MatingPool1=MatingPool1.reshape(-1)
                    Offspring = OperatorGA(Population.sort_index(MatingPool1.astype(int)), xl, xu)
                else:
                    #利用history_pop中的历史个体结合Transformer网络向前预测一代
                    print("Transformer训练")
                    net.train(history_pop_Population)
                    #网络生成
                    result=net.predict_future(history_pop_Population)
                    result=result.cpu() 
                    #形状变换
                    last_num=result.shape[0]
                    new_EXA=result[last_num-1,:]
                    new_EXA=torch.reshape(new_EXA, (dim_n, n_var+n_obj))  #EXA表示生成的子代群体
                    Offspring=new_EXA.numpy()
                    Offspring=Offspring[:,0:n_var]
                    #边界控制：
                    Offspring=fixnew(Offspring,xl,xu)
                    
            if(train_i%6==0):
                #清空历史
                parent_dis1=history_pop_Population[-1].decs
                parent_dis2=history_pop_Archive[-1].decs
                history_pop_Population=[]
                history_pop_Archive=[]
            
            #评价次数
            Offspring_r=np.zeros((Offspring.shape[0],n_obj))
            for k in range (Offspring.shape[0]):
                temp_x=Offspring[k,0:n_var]
                temp_new_point=np.copy(temp_x)
                t_reult=eval(fname)(temp_new_point)
                FEx=FEx+1
                t_reult=t_reult.reshape(1,n_obj)   
                Offspring_r[k,0:n_obj]=t_reult
            
            Offspring=pop_class(Offspring,Offspring_r)

            Population,CrowdDis1 = EnvironmentalSelection(Population.com_pop(Offspring),Popsize)
            #存储历史种群：
            if(train_i%6>0):
                history_pop_Population.append(Population)
            Archive,CrowdDis2 = ArchiveUpdate3(Archive.com_pop(Offspring),Popsize,eps,FEx/maxFEx,i)
            if(train_i%6>0):
                history_pop_Archive.append(Archive)
            train_i=train_i+1
            #万轮存储
            '''
            ps=Archive.decs
            pf=Archive.objs
            
            if(i<=22):
                PS=sio.loadmat(TruePSPF_path)['PS']
                PF=sio.loadmat(TruePSPF_path)['PF'] 
                S_CR=CR_calculation(ps,PS);     
                S_IGDx=IGD_calculation(ps,PS)
                S_IGDf=IGD_calculation(pf,PF)
            else:
                PSS=sio.loadmat(TruePSPF_path)['PSS']
                PF=sio.loadmat(TruePSPF_path)['PF']
                S_CR=CR_calculation(ps,PSS);     
                S_IGDx=IGD_calculation(ps,PSS)
                S_IGDf=IGD_calculation(pf,PF)    

            S_PSP=S_CR/S_IGDx
            IGDx.append(S_IGDx)
            IGDf.append(S_IGDf)
            CR.append(S_CR)
            PSP.append(S_PSP)
            '''
        
        ps=Archive.decs
        pf=Archive.objs

        if(i<=22):
            PS=sio.loadmat(TruePSPF_path)['PS']
            PF=sio.loadmat(TruePSPF_path)['PF'] 
            S_CR=CR_calculation(ps,PS)  
            S_IGDx=IGD_calculation(ps,PS)
            S_IGDf=IGD_calculation(pf,PF)
        else:
            PSS=sio.loadmat(TruePSPF_path)['PSS']
            PF=sio.loadmat(TruePSPF_path)['PF']
            S_CR=CR_calculation(ps,PSS)    
            S_IGDx=IGD_calculation(ps,PSS)
            S_IGDf=IGD_calculation(pf,PF)    

        S_PSP=S_CR/S_IGDx
        IGDx.append(S_IGDx)
        IGDf.append(S_IGDf)
        CR.append(S_CR)
        PSP.append(S_PSP)
        
    
    m_IGDx=np.mean(IGDx)
    print("IGDx",m_IGDx)
    m_IGDf=np.mean(IGDf)
    print("IGDf",m_IGDf)
    m_CR=np.mean(CR)
    print("CR",m_CR)
    m_PSP=np.mean(PSP)
    print("PSP",m_PSP)
    
    file_name='./result/offspringDistribution/'+fname+'_result.mat'
    #sio.savemat(file_name, {'IGDx': IGDx,'m_IGDx':m_IGDx,'IGDf':IGDf,'m_IGDf':m_IGDf,'CR':CR,'m_CR':m_CR,'PSP':PSP,'m_PSP':m_PSP,'ps':ps,'pf':pf})
    print("PSP",PSP)
    print("IGDx",IGDx)
    print("IGDf",IGDf)
    sio.savemat(file_name, {'IGDx': IGDx,'IGDf':IGDf,'CR':CR,'PSP':PSP, 'ps':ps,'pf':pf,'parent_dis1':parent_dis1,'parent_dis2':parent_dis2})
        




