from Init import *
import scipy.io as sio
import os
from Functions.Indicator_calculation import*
from Functions.ArchiveUpdate import *
from Functions.TournamentSelection import *
from OperatorGA import *
from ModelClass import *
import time
import mat73
runs=20 #运行次数
pi_value = np.pi
eps=0.1
p = 0.5
history_length=8
batch_size = 2
epoch=300
lr=0.01
dropout=0.5

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

def paraConfig(i):
    # data CEC2020数据集
    if (i == 1):
        fname = 'MMF1'
        n_var = 2
        n_obj = 2
        xl = np.array([1, -1])
        xu = np.array([3, 1])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF1_Reference_PSPF_data.mat'
    elif (i == 2):
        fname = 'MMF1_e'
        n_var = 2
        n_obj = 2
        xl = np.array([1, -20])
        xu = np.array([3, 20])
        repoint = np.array([[1.1, 1.1]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF1_e_Reference_PSPF_data.mat'
    elif (i == 3):
        fname = 'MMF2'
        n_var = 2
        n_obj = 2
        xl = np.array([0, 0])
        xu = np.array([1, 2])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF2_Reference_PSPF_data.mat'
    elif (i == 4):
        fname = 'MMF4'
        n_var = 2
        n_obj = 2
        xl = np.array([-1, 0])
        xu = np.array([1, 2])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF4_Reference_PSPF_data.mat'
    elif (i == 5):
        fname = 'MMF5'
        n_var = 2
        n_obj = 2
        xl = np.array([1, -1])
        xu = np.array([3, 3])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF5_Reference_PSPF_data.mat'
    elif (i == 6):
        fname = 'MMF7'
        n_var = 2
        n_obj = 2
        xl = np.array([1, -1])
        xu = np.array([3, 1])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF7_Reference_PSPF_data.mat'
    elif (i == 7):
        fname = 'MMF8'
        n_var = 2
        n_obj = 2
        xl = np.array([-pi_value, 0])
        xu = np.array([pi_value, 9])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF8_Reference_PSPF_data.mat'
    elif (i == 8):
        fname = 'MMF10'
        n_var = 2
        n_obj = 2
        xl = np.array([0.1, 0.1])
        xu = np.array([1.1, 1.1])
        repoint = np.array([[1.21, 13.2]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF10_Reference_PSPF_data.mat'
    elif (i == 9):
        fname = 'MMF10_l'
        n_var = 2
        n_obj = 2
        xl = np.array([0.1, 0.1])
        xu = np.array([1.1, 1.1])
        repoint = np.array([[1.21, 13.2]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF10_l_Reference_PSPF_data.mat'
    elif (i == 10):
        fname = 'MMF11'
        n_var = 2
        n_obj = 2
        xl = np.array([0.1, 0.1])
        xu = np.array([1.1, 1.1])
        repoint = np.array([[1.21, 15.4]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF11_Reference_PSPF_data.mat'
    elif (i == 11):
        fname = 'MMF11_l'
        n_var = 2
        n_obj = 2
        xl = np.array([0.1, 0.1])
        xu = np.array([1.1, 1.1])
        repoint = np.array([[1.21, 15.4]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF11_l_Reference_PSPF_data.mat'
    elif (i == 12):
        fname = 'MMF12'
        n_var = 2
        n_obj = 2
        xl = np.array([0, 0])
        xu = np.array([1, 1])
        repoint = np.array([[1.54, 1.1]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF12_Reference_PSPF_data.mat'
    elif (i == 13):
        fname = 'MMF12_l'
        n_var = 2
        n_obj = 2
        xl = np.array([0, 0])
        xu = np.array([1, 1])
        repoint = np.array([[1.54, 1.1]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF12_l_Reference_PSPF_data.mat'
    elif (i == 14):
        fname = 'MMF13'
        n_var = 3
        n_obj = 2
        xl = np.array([0.1, 0.1, 0.1])
        xu = np.array([1.1, 1.1, 1.1])
        repoint = np.array([[1.54, 15.4]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF13_Reference_PSPF_data.mat'
    elif (i == 15):
        fname = 'MMF13_l'
        n_var = 3
        n_obj = 2
        xl = np.array([0.1, 0.1, 0.1])
        xu = np.array([1.1, 1.1, 1.1])
        repoint = np.array([[1.54, 15.4]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF13_l_Reference_PSPF_data.mat'
    elif (i == 16):
        fname = 'MMF14'
        n_var = 3
        n_obj = 3
        xl = np.array([0, 0, 0])
        xu = np.array([1, 1, 1])
        repoint = np.array([[2.2, 2.2, 2.2]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF14_Reference_PSPF_data.mat'
        # PS_number=2
    elif (i == 17):
        fname = 'MMF14_a'
        n_var = 3
        n_obj = 3
        xl = np.array([0, 0, 0])
        xu = np.array([1, 1, 1])
        repoint = np.array([[2.2, 2.2, 2.2]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF14_a_Reference_PSPF_data.mat'
    elif (i == 18):
        fname = 'MMF15'
        n_var = 3
        n_obj = 3
        xl = np.array([0, 0, 0])
        xu = np.array([1, 1, 1])
        repoint = np.array([[2.5, 2.5, 2.5]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF15_Reference_PSPF_data.mat'
    elif (i == 19):
        fname = 'MMF15_l'
        n_var = 3
        n_obj = 3
        xl = np.array([0, 0, 0])
        xu = np.array([1, 1, 1])
        repoint = np.array([[2.5, 2.5, 2.5]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF15_l_Reference_PSPF_data.mat'
    elif (i == 20):
        fname = 'MMF15_a'
        n_var = 3
        n_obj = 3
        xl = np.array([0, 0, 0])
        xu = np.array([1, 1, 1])
        repoint = np.array([[2.5, 2.5, 2.5]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF15_a_Reference_PSPF_data.mat'
    elif (i == 21):
        fname = 'MMF15_a_l'
        n_var = 3
        n_obj = 3
        xl = np.array([0, 0, 0])
        xu = np.array([1, 1, 1])
        repoint = np.array([[2.5, 2.5, 2.5]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF15_a_l_Reference_PSPF_data.mat'
    elif (i == 22):
        fname = 'MMF16_l1'
        n_var = 3
        n_obj = 3
        xl = np.array([0, 0, 0])
        xu = np.array([1, 1, 1])
        repoint = np.array([[2.5, 2.5, 2.5]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF16_l1_Reference_PSPF_data.mat'
    elif (i == 23):
        fname = 'MMF16_l2'
        n_var = 3
        n_obj = 3
        xl = np.array([0, 0, 0])
        xu = np.array([1, 1, 1])
        repoint = np.array([[2.5, 2.5, 2.5]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF16_l2_Reference_PSPF_data.mat'
    elif (i == 24):
        fname = 'MMF16_l3'
        n_var = 3
        n_obj = 3
        xl = np.array([0, 0, 0])
        xu = np.array([1, 1, 1])
        repoint = np.array([[2.5, 2.5, 2.5]])
        TruePSPF_path = './TruePS_PF/CEC2020/MMF16_l3_Reference_PSPF_data.mat'

    # IDMP数据集
    elif (i == 25):
        fname = 'IDMPM2T1'
        n_var = 2
        n_obj = 2
        xl = np.array([-1, -1])
        xu = np.array([1, 1])
        repoint = np.array([[1, 1]])
        TruePSPF_path = './TruePS_PF/IDMP/IDMPM2T1Reference.mat'
    elif (i == 26):
        fname = 'IDMPM2T2'
        n_var = 2
        n_obj = 2
        xl = np.array([-1, -1])
        xu = np.array([1, 1])
        repoint = np.array([[1, 1]])
        TruePSPF_path = './TruePS_PF/IDMP/IDMPM2T2Reference.mat'
    elif (i == 27):
        fname = 'IDMPM2T3'
        n_var = 2
        n_obj = 2
        xl = np.array([-1, -1])
        xu = np.array([1, 1])
        repoint = np.array([[1, 1]])
        TruePSPF_path = './TruePS_PF/IDMP/IDMPM2T3Reference.mat'
    elif (i == 28):
        fname = 'IDMPM2T4'
        n_var = 2
        n_obj = 2
        xl = np.array([-1, -1])
        xu = np.array([1, 1])
        repoint = np.array([[1, 1]])
        TruePSPF_path = './TruePS_PF/IDMP/IDMPM2T4Reference.mat'
    elif (i == 29):
        fname = 'IDMPM3T1'
        n_var = 3
        n_obj = 3
        xl = np.array([-1, -1, -1])
        xu = np.array([1, 1, 1])
        repoint = np.array([[1, 1, 1]])
        TruePSPF_path = './TruePS_PF/IDMP/IDMPM3T1Reference.mat'
    elif (i == 30):
        fname = 'IDMPM3T2'
        n_var = 3
        n_obj = 3
        xl = np.array([-1, -1, -1])
        xu = np.array([1, 1, 1])
        repoint = np.array([[1, 1, 1]])
        TruePSPF_path = './TruePS_PF/IDMP/IDMPM3T2Reference.mat'
    elif (i == 31):
        fname = 'IDMPM3T3'
        n_var = 3
        n_obj = 3
        xl = np.array([-1, -1, -1])
        xu = np.array([1, 1, 1])
        repoint = np.array([[1, 1, 1]])
        TruePSPF_path = './TruePS_PF/IDMP/IDMPM3T3Reference.mat'
    elif (i == 32):
        fname = 'IDMPM3T4'
        n_var = 3
        n_obj = 3
        xl = np.array([-1, -1, -1])
        xu = np.array([1, 1, 1])
        repoint = np.array([[1, 1, 1]])
        TruePSPF_path = './TruePS_PF/IDMP/IDMPM3T4Reference.mat'
    elif (i == 33):
        fname = 'IDMPM4T1'
        n_var = 4
        n_obj = 4
        xl = np.array([-1, -1, -1, -1])
        xu = np.array([1, 1, 1, 1])
        repoint = np.array([[1, 1, 1, 1]])
        TruePSPF_path = './TruePS_PF/IDMP/IDMPM4T1Reference.mat'
    elif (i == 34):
        fname = 'IDMPM4T2'
        n_var = 4
        n_obj = 4
        xl = np.array([-1, -1, -1, -1])
        xu = np.array([1, 1, 1, 1])
        repoint = np.array([[1, 1, 1, 1]])
        TruePSPF_path = './TruePS_PF/IDMP/IDMPM4T2Reference.mat'
    elif (i == 35):
        fname = 'IDMPM4T3'
        n_var = 4
        n_obj = 4
        xl = np.array([-1, -1, -1, -1])
        xu = np.array([1, 1, 1, 1])
        repoint = np.array([[1, 1, 1, 1]])
        TruePSPF_path = './TruePS_PF/IDMP/IDMPM4T3Reference.mat'
    elif (i == 36):
        fname = 'IDMPM4T4'
        n_var = 4
        n_obj = 4
        xl = np.array([-1, -1, -1, -1])
        xu = np.array([1, 1, 1, 1])
        repoint = np.array([[1, 1, 1, 1]])
        TruePSPF_path = './TruePS_PF/IDMP/IDMPM4T4Reference.mat'

    # data MMMOP
    elif (i == 37):
        fname = 'MMMOP1A'
        n_var = 3
        n_obj = 2
        xl = np.array([0, 0, 0])
        xu = np.array([1, 1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP1A_PFPS.mat'
    elif i == 38:
        fname = 'MMMOP1B'
        n_obj = 3
        n_var = 7
        xl = np.array([0, 0, 0, 0, 0, 0, 0])
        xu = np.array([1, 1, 1, 1, 1, 1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP1B_PFPS.mat'
    elif i == 39:
        fname = 'MMMOP2A'
        n_obj = 2
        n_var = 3
        xl = np.array([0, 0, 0])
        xu = np.array([1, 1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP2A_PFPS.mat'
    elif i == 40:
        fname = 'MMMOP2B'
        n_obj = 3
        n_var = 7
        xl = np.array([0, 0, 0, 0, 0, 0, 0])
        xu = np.array([1, 1, 1, 1, 1, 1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP2B_PFPS.mat'
    elif i == 41:
        fname = 'MMMOP3A'
        n_obj = 2
        n_var = 2
        xl = np.array([0, 0])
        xu = np.array([1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP3A_PFPS.mat'
    elif i == 42:
        fname = 'MMMOP3B'
        n_obj = 3
        n_var = 7
        xl = np.array([0, 0, 0, 0, 0, 0, 0])
        xu = np.array([1, 1, 1, 1, 1, 1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP3B_PFPS.mat'
    elif i == 43:
        fname = 'MMMOP3C'
        n_obj = 2
        n_var = 6
        xl = np.array([0, 0, 0, 0, 0, 0])
        xu = np.array([1, 1, 1, 1, 1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP3C_PFPS.mat'
    elif i == 44:
        fname = 'MMMOP3D'
        n_obj = 3
        n_var = 7
        xl = np.array([0, 0, 0, 0, 0, 0, 0])
        xu = np.array([1, 1, 1, 1, 1, 1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP3D_PFPS.mat'
    elif i == 45:
        fname = 'MMMOP4A'
        n_obj = 2
        n_var = 2
        xl = np.array([0, 0])
        xu = np.array([1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP4A_PFPS.mat'
    elif i == 46:
        fname = 'MMMOP4B'
        n_obj = 3
        n_var = 7
        xl = np.array([0, 0, 0, 0, 0, 0, 0])
        xu = np.array([1, 1, 1, 1, 1, 1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP4B_PFPS.mat'
    elif i == 47:
        fname = 'MMMOP4C'
        n_obj = 2
        n_var = 6
        xl = np.array([0, 0, 0, 0, 0, 0])
        xu = np.array([1, 1, 1, 1, 1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP4C_PFPS.mat'
    elif i == 48:
        fname = 'MMMOP4D'
        n_obj = 3
        n_var = 7
        xl = np.array([0, 0, 0, 0, 0, 0, 0])
        xu = np.array([1, 1, 1, 1, 1, 1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP4D_PFPS.mat'
    elif i == 49:
        fname = 'MMMOP5A'
        n_obj = 2
        n_var = 2
        xl = np.array([0, 0])
        xu = np.array([1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP5A_PFPS.mat'
    elif i == 50:
        fname = 'MMMOP5B'
        n_obj = 3
        n_var = 7
        xl = np.array([0, 0, 0, 0, 0, 0, 0])
        xu = np.array([1, 1, 1, 1, 1, 1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP5B_PFPS.mat'
    elif i == 51:
        fname = 'MMMOP5C'
        n_obj = 2
        n_var = 6
        xl = np.array([0, 0, 0, 0, 0, 0])
        xu = np.array([1, 1, 1, 1, 1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP5C_PFPS.mat'
    elif i == 52:
        fname = 'MMMOP5D'
        n_obj = 3
        n_var = 7
        xl = np.array([0, 0, 0, 0, 0, 0, 0])
        xu = np.array([1, 1, 1, 1, 1, 1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP5D_PFPS.mat'
    elif i == 53:
        fname = 'MMMOP6A'
        n_obj = 2
        n_var = 2
        xl = np.array([0, 0])
        xu = np.array([1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP6A_PFPS.mat'
    elif i == 54:
        fname = 'MMMOP6B'
        n_obj = 3
        n_var = 4
        xl = np.array([0, 0, 0, 0])
        xu = np.array([1, 1, 1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP6B_PFPS.mat'
    elif i == 55:
        fname = 'MMMOP6C'
        n_obj = 2
        n_var = 4
        xl = np.array([0, 0, 0, 0])
        xu = np.array([1, 1, 1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP6C_PFPS.mat'
    elif i == 56:
        fname = 'MMMOP6D'
        n_obj = 3
        n_var = 5
        xl = np.array([0, 0, 0, 0, 0])
        xu = np.array([1, 1, 1, 1, 1])
        TruePSPF_path = './TruePS_PF/MMMOP/MMMOP6D_PFPS.mat'
    # data mapfunctiontruePSPF
    elif i == 57:
        fname = 'mapfunction'
        n_var = 2
        n_obj = 4
        xl = np.array([0, 0])
        xu = np.array([100, 100])
        TruePSPF_path = './TruePS_PF/mapfunctiontruePSPF.mat'

    # data SMMOP
    elif i == 58:
        fname = 'SMMOP1'
        n_obj = 2
        n_var = 100
        xl = np.concatenate([np.zeros(n_obj - 1) + 0, np.zeros(n_var - n_obj + 1) - 1])
        xu = np.concatenate([np.zeros(n_obj - 1) + 1, np.zeros(n_var - n_obj + 1) + 2])
        TruePSPF_path = './TruePS_PF/SMMOP/SMMOP1_truePSPF.mat'
    elif i == 59:
        fname = 'SMMOP2'
        n_obj = 2
        n_var = 100
        xl = np.concatenate([np.zeros(n_obj - 1) + 0, np.zeros(n_var - n_obj + 1) - 1])
        xu = np.concatenate([np.zeros(n_obj - 1) + 1, np.zeros(n_var - n_obj + 1) + 2])
        TruePSPF_path = './TruePS_PF/SMMOP/SMMOP2_truePSPF.mat'
    elif i == 60:
        fname = 'SMMOP3'
        n_obj = 2
        n_var = 100
        xl = np.concatenate([np.zeros(n_obj - 1) + 0, np.zeros(n_var - n_obj + 1) - 1])
        xu = np.concatenate([np.zeros(n_obj - 1) + 1, np.zeros(n_var - n_obj + 1) + 2])
        TruePSPF_path = './TruePS_PF/SMMOP/SMMOP3_truePSPF.mat'
    elif i == 61:
        fname = 'SMMOP4'
        n_obj = 2
        n_var = 100
        xl = np.concatenate([np.zeros(n_obj - 1) + 0, np.zeros(n_var - n_obj + 1) - 1])
        xu = np.concatenate([np.zeros(n_obj - 1) + 1, np.zeros(n_var - n_obj + 1) + 2])
        TruePSPF_path = './TruePS_PF/SMMOP/SMMOP4_truePSPF.mat'
    elif i == 62:
        fname = 'SMMOP5'
        n_obj = 2
        n_var = 100
        xl = np.concatenate([np.zeros(n_obj - 1) + 0, np.zeros(n_var - n_obj + 1) - 1])
        xu = np.concatenate([np.zeros(n_obj - 1) + 1, np.zeros(n_var - n_obj + 1) + 2])
        TruePSPF_path = './TruePS_PF/SMMOP/SMMOP5_truePSPF.mat'
    elif i == 63:
        fname = 'SMMOP6'
        n_obj = 2
        n_var = 100
        xl = np.concatenate([np.zeros(n_obj - 1) + 0, np.zeros(n_var - n_obj + 1) - 1])
        xu = np.concatenate([np.zeros(n_obj - 1) + 1, np.zeros(n_var - n_obj + 1) + 2])
        TruePSPF_path = './TruePS_PF/SMMOP/SMMOP6_truePSPF.mat'
    elif i == 64:
        fname = 'SMMOP7'
        n_obj = 2
        n_var = 100
        xl = np.concatenate([np.zeros(n_obj - 1) + 0, np.zeros(n_var - n_obj + 1) - 1])
        xu = np.concatenate([np.zeros(n_obj - 1) + 1, np.zeros(n_var - n_obj + 1) + 2])
        TruePSPF_path = './TruePS_PF/SMMOP/SMMOP7_truePSPF.mat'
    elif i == 65:
        fname = 'SMMOP8'
        n_obj = 2
        n_var = 100
        xl = np.concatenate([np.zeros(n_obj - 1) + 0, np.zeros(n_var - n_obj + 1) - 1])
        xu = np.concatenate([np.zeros(n_obj - 1) + 1, np.zeros(n_var - n_obj + 1) + 2])
        TruePSPF_path = './TruePS_PF/SMMOP/SMMOP8_truePSPF.mat'
    return  fname,n_var,n_obj,xl,xu,TruePSPF_path

datasetSeed={
    # "CEC2020":[1,25],
    # "IDMP":[25,37],
    # "MMMOP":[37,57],
    # "SMMOP":[58,66]
    # "trainBetter":[19,22,15,9,11,13,15,21]
    "trainBetter":[6]
}
result_print={}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
else:
    device = torch.device("cpu")

for datasetName,questionBand in datasetSeed.items():
    result_print[datasetName] = {}
    outputPath=os.path.join(r'./results/',datasetName)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    if len(questionBand)==2:
        questionIDs=[i for i in range(questionBand[0], questionBand[1])]
    else:
        questionIDs=questionBand
    for i in questionIDs:
        print("函数：", i)
        hyp = []
        IGDx = []
        IGDf = []
        CR = []
        PSP = []
        RTime=[]
        '''
        fname:优化目标函数名
        n_var：决策空间维度
        n_obj：目标空间维度
        xl, xu：决策变量上下界
        '''
        fname, n_var, n_obj, xl, xu, TruePSPF_path = paraConfig(i)
        for j in range(1, runs + 1):
            print("运行次数：", j)
            # epoch = 300
            # lr = 0.02
            # dropout = 0.5
            start = time.time()
            Popsize = 400 * n_var
            maxFEx = 20000 * n_var
            chrom, obj = Init(Popsize, n_var, n_obj, fname, xl, xu)
            Population = pop_class(chrom, obj)
            _, CrowdDis1 = EnvironmentalSelection(Population, Popsize)
            Archive, CrowdDis2 = ArchiveUpdate(Population, Popsize, eps, 0, i)

            train_i = 1
            FEx = Popsize

            # net = STGSAGE(lr, epoch, n_var, n_obj, fname, Popsize)
            net=ControllerModel(lr, epoch, n_var, n_obj, fname, Popsize,history_length-1,dropout=dropout)

            history_pop_Population = []  # transformer的作用是利用历史种群推导出新种群，history_pop用于存储历史种群
            # history_pop_Archive=[]
            history_pop_Population.append(Population)
            # history_pop_Archive.append(Archive)
            print("train_i", train_i)
            print("FEx:", FEx)
            train_i = train_i + 1
            print("train_i", train_i)
            while (FEx < maxFEx):
                if FEx >= maxFEx * 0.5 and np.random.rand() < p:  # 注意随机数定义的不同matlab函数中的rand取值范围是(0,1),而python中的是[0,1)
                    # if(train_i%7>0):
                    MatingPool2 = TournamentSelection(2, round(Popsize), -CrowdDis2)
                    MatingPool2 = MatingPool2.reshape(-1)
                    Offspring = OperatorGA(Archive.sort_index(MatingPool2.astype(int)), xl, xu)
                    # else:
                    # #利用history_pop中的历史个体结合Transformer网络向前预测一代
                    # print("DGNN训练")
                    # net.train(history_pop_Archive)
                    # #网络生成
                    # result=net.predict_offspring(history_pop_Archive)
                    # result=result.cpu()
                    # #形状变换
                    # Offspring=result.detach().numpy()
                    # Offspring=Offspring[:,0:n_var]
                    # #边界控制：
                    # Offspring=fixnew(Offspring,xl,xu)
                else:
                    if (train_i % history_length > 0):
                        MatingPool1 = TournamentSelection(2, round(Popsize), -CrowdDis1)
                        MatingPool1 = MatingPool1.reshape(-1)
                        Offspring = OperatorGA(Population.sort_index(MatingPool1.astype(int)), xl, xu)
                    else:
                        print("Gsage训练")
                        net.train(history_pop_Population)
                        # 网络生成
                        result = net.predict_offspring(history_pop_Population)
                        result = result.cpu()
                        # 形状变换
                        Offspring = result.detach().numpy()
                        Offspring = Offspring[:, 0:n_var]
                        # 边界控制：
                        Offspring = fixnew(Offspring, xl, xu)

                if (train_i % history_length == 0):
                    # 清空历史
                    history_pop_Population = []
                    # history_pop_Archive=[]
                # 评价次数
                Offspring_r = np.zeros((Offspring.shape[0], n_obj))
                for k in range(Offspring.shape[0]):
                    temp_x = Offspring[k, 0:n_var]
                    temp_new_point = np.copy(temp_x)
                    t_reult = eval(fname)(temp_new_point)
                    FEx = FEx + 1
                    t_reult = t_reult.reshape(1, n_obj)
                    Offspring_r[k, 0:n_obj] = t_reult

                Offspring = pop_class(Offspring, Offspring_r)

                Population, CrowdDis1 = EnvironmentalSelection(Population.com_pop(Offspring), Popsize)
                # 存储历史种群：
                if (train_i % history_length > 0):
                    history_pop_Population.append(Population)
                # print("Offspring:",Offspring.length)
                Archive, CrowdDis2 = ArchiveUpdate(Archive.com_pop(Offspring), Popsize, eps, FEx / maxFEx, i)
                # if(train_i%7>0):
                #     history_pop_Archive.append(Archive)
                train_i = train_i + 1
                print("FEx:", FEx)
                print("train_i", train_i)
                # 万轮存储
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

            ps = Archive.decs
            pf = Archive.objs

            # train_data = np.array([history_pop_Population[0].pop, history_pop_Population[1].pop,history_pop_Population[2].pop,history_pop_Population[3].pop,history_pop_Population[4].pop], history_pop_Population[5].pop)
            # np.save('./Data/train_data.npy', history_pop_Population)

            if (i <= 24):
                PS = sio.loadmat(TruePSPF_path)['PS']
                PF = sio.loadmat(TruePSPF_path)['PF']
                S_CR = CR_calculation(ps, PS)
                S_IGDx = IGD_calculation(ps, PS)
                S_IGDf = IGD_calculation(pf, PF)
            elif (i <= 56):
                PSS = sio.loadmat(TruePSPF_path)['PSS']
                PF = sio.loadmat(TruePSPF_path)['PF']
                S_CR = CR_calculation(ps, PSS)
                S_IGDx = IGD_calculation(ps, PSS)
                S_IGDf = IGD_calculation(pf, PF)
            else:
                PS = mat73.loadmat(TruePSPF_path)['PS']
                PF = mat73.loadmat(TruePSPF_path)['PF']
                S_CR = CR_calculation(ps, PS)
                S_IGDx = IGD_calculation(ps, PS)
                S_IGDf = IGD_calculation(pf, PF)

            S_PSP = S_CR / S_IGDx
            IGDx.append(S_IGDx)
            IGDf.append(S_IGDf)
            CR.append(S_CR)
            PSP.append(S_PSP)
            end = time.time()
            rtime = end - start
            RTime.append(rtime)

        m_IGDx = np.mean(IGDx)
        print(fname,"IGDx(mean)", m_IGDx)
        m_IGDf = np.mean(IGDf)
        print(fname,"IGDf(mean)", m_IGDf)
        m_CR = np.mean(CR)
        print(fname,"CR(mean)", m_CR)
        m_PSP = np.mean(PSP)
        print(fname,"PSP(mean)", m_PSP)
        m_time=np.mean(RTime)
        print(fname,"Time(mean)", m_time)
        result_print[datasetName][fname]={'IGDx': m_IGDx, 'IGDf': m_IGDf, 'CR': m_CR, 'PSP': m_PSP, 'time': m_time}
        file_name = os.path.join(outputPath,fname + '.mat')
        # print(datasetName,"PSP", PSP)
        # print(datasetName,"IGDx", IGDx)
        # print(datasetName,"IGDf", IGDf)
        # print(datasetName,"time", rtime)
        sio.savemat(file_name, {'IGDx': IGDx, 'IGDf': IGDf, 'CR': CR, 'PSP': PSP, 'ps': ps, 'pf': pf, 'time': RTime})

print(result_print)