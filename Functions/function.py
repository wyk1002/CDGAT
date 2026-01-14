import numpy as np
from math import sin, sqrt, pi, cos
from scipy.spatial.distance import cdist
#from matplotlib.patches import Polygon
from shapely.geometry import Polygon

#CEC2019数据集

def MMF1(x):
    # 1<=x1<=3 -1<=x2<=1
    y = np.zeros(2)
    x[0] = abs((x[0]-2))
    y[0] = x[0]
    y[1] = 1.0 - sqrt(x[0]) + 2.0*(x[1]-sin(6*pi*x[0]+pi))**2
    return y

def MMF1_e(x):
    # 1<=x1<=3, -1<=x2<=1
    # x: number_of_point * number_of_decision_var
    # y: number_of_point * number_of_objective
    x=x.reshape(1, 2)

    left_index = np.where(x[:, 0] < 2)
    right_index = np.where(x[:, 0] >= 2)
    
    y = np.zeros((2, x.shape[0]))
    
    y[0, left_index] = 2 - x[left_index, 0]
    y[0, right_index] = x[right_index, 0] - 2
    
    y[1, left_index] = 1.0 - np.sqrt(2 - x[left_index, 0]) + 2.0 * (x[left_index, 1] - np.sin(6 * np.pi * (2 - x[left_index, 0]) + np.pi))**2
    y[1, right_index] = 1.0 - np.sqrt(x[right_index, 0] - 2) + 2.0 * (x[right_index, 1] - np.exp(x[right_index, 0]) * np.sin(6 * np.pi * (x[right_index, 0] - 2) + np.pi))**2
    
    y=y.reshape(-1)
    return y

def MMF1_z(x):
    x=x.reshape(1, 2)

    left_index = np.where(x[:,0] < 2)
    right_index = np.where(x[:,0] >= 2)
    
    y = np.zeros((x.shape[0], 2))
    
    y[left_index, 0] = 2 - x[left_index, 0]
    y[right_index, 0] = x[right_index, 0] - 2
    
    y[left_index, 1] = 1.0 - np.sqrt(2 - x[left_index, 0]) + 2.0 * (x[left_index, 1] - np.sin(6 * np.pi * (2 - x[left_index, 0]) + np.pi))**2
    y[right_index, 1] = 1.0 - np.sqrt(x[right_index, 0] - 2) + 2.0 * (x[right_index, 1] - np.sin(2 * np.pi * (x[right_index, 0] - 2) + np.pi))**2
    y=y.T
    y=y.reshape(-1)
    return y

def MMF2(x):
    f = np.zeros(2)
    if x[1] > 1:
        x[1] = x[1] - 1
    f[0] = x[0]
    y2 = x[1] - x[0]**0.5
    f[1] = 1.0 - np.sqrt(x[0]) + 2*((4*y2**2)-2*np.cos(20*y2*np.pi/np.sqrt(2))+2)
    return f

def MMF3(x):
    f = np.zeros(2)
    f[0] = x[0]
    if 0 <= x[1] <= 0.5:
        y2 = x[1] - x[0] ** 0.5
    elif 0.5 < x[1] < 1 and 0 <= x[0] <= 0.25:
        y2 = x[1] - 0.5 - x[0] ** 0.5
    elif 0.5 < x[1] < 1 and x[0] > 0.25:
        y2 = x[1] - x[0] ** 0.5
    elif 1 <= x[1] <= 1.5:
        y2 = x[1] - 0.5 - x[0] ** 0.5
    f[1] = 1.0 - x[0] ** 0.5 + 2 * ((4 * y2 ** 2) - 2 * np.cos(20 * y2 * np.pi / np.sqrt(2)) + 2)
    
    return f

def MMF4(x):
    f = np.zeros(2)
    if x[1] > 1:
        x[1] = x[1] - 1
    f[0] = abs(x[0])
    f[1] = 1.0 - x[0]**2 + 2 * (x[1] - np.sin(np.pi * abs(x[0])))**2
    
    return f

def MMF5(x):
    Obj = np.zeros(2)
    if x[1] > 1:
        x[1] = x[1] - 2
    Obj[0] = abs(x[0] - 2)
    Obj[1] = 1.0 - np.sqrt(abs(x[0] - 2)) + 2.0 * (x[1] - np.sin(6 * np.pi * abs(x[0] - 2) + np.pi))**2
    
    return Obj

def MMF6(x):
    Obj = np.zeros(2)
    
    if (x[1] > -1 and x[1] <= 0) and (((x[0] > 7/6 and x[0] <= 8/6)) or (x[0] > 9/6 and x[0] <= 10/6) or (x[0] > 11/6 and x[0] <= 2)):
        x[1] = x[1]
    elif (x[1] > -1 and x[1] <= 0) and ((x[0] > 2 and x[0] <= 13/6) or (x[0] > 14/6 and x[0] <= 15/6) or (x[0] > 16/6 and x[0] <= 17/6)):
        x[1] = x[1]
    elif (x[1] > 1 and x[1] <= 2) and ((x[0] > 1 and x[0] <= 7/6) or (x[0] > 4/3 and x[0] <= 3/2) or (x[0] > 5/3 and x[0] <= 11/6)):
        x[1] = x[1] - 1
    elif (x[1] > 1 and x[1] <= 2) and ((x[0] > 13/6 and x[0] <= 14/6) or (x[0] > 15/6 and x[0] <= 16/6) or (x[0] > 17/6 and x[0] <= 3)):
        x[1] = x[1] - 1
    elif (x[1] > 0 and x[1] <= 1) and ((x[0] > 1 and x[0] <= 7/6) or (x[0] > 4/3 and x[0] <= 3/2) or (x[0] > 5/3 and x[0] <= 11/6)
                                    or (x[0] > 13/6 and x[0] <= 14/6) or (x[0] > 15/6 and x[0] <= 16/6) or (x[0] > 17/6 and x[0] <= 3)):
        x[1] = x[1]
    elif (x[1] > 0 and x[1] <= 1) and ((x[0] > 7/6 and x[0] <= 8/6) or (x[0] > 9/6 and x[0] <= 10/6) or (x[0] > 11/6 and x[0] <= 2)
                                      or (x[0] > 2 and x[0] <= 13/6) or (x[0] > 14/6 and x[0] <= 15/6) or (x[0] > 16/6 and x[0] <= 17/6)):
        x[1] = x[1] - 1
    
    Obj[0] = abs(x[0] - 2)
    Obj[1] = 1.0 - np.sqrt(abs(x[0] - 2)) + 2.0 * (x[1] - np.sin(6 * np.pi * abs(x[0] - 2) + np.pi))**2
    
    return Obj

def MMF7(x):
    Obj = np.zeros(2)
    
    x[0] = abs(x[0] - 2)
    Obj[0] = x[0]
    Obj[1] = 1.0 - np.sqrt(x[0]) + (x[1] - (0.3 * (x[0] ** 2) * np.cos(24 * np.pi * x[0] + 4 * np.pi) + 0.6 * x[0]) * np.sin(6 * np.pi * x[0] + np.pi)) ** 2
    
    return Obj

def MMF8(x):
    #print("调用函数MMF8")
    #print("x1",x)
    Obj = np.zeros(2)
    #print("x2",x)
    if x[1] > 4:
        x[1] = x[1] - 4
    #print("x3",x)
    Obj[0] = np.sin(np.abs(x[0]))
    #print("Obj[0]",Obj[0])
    Obj[1] = np.sqrt(1.0 - np.sin(np.abs(x[0]))**2) + 2.0 * (x[1] - (np.sin(np.abs(x[0])) + np.abs(x[0])))**2
    #print("Obj[1]",Obj[1])
    return Obj

def MMF9(x):
    # 0.1 < x1 <= 1.1, 0.1 <= x2 <= 1.1, g(x2) > 0
    
    y = np.zeros(2)
    num_of_peak = 2
    
    temp1 = (np.sin(num_of_peak * np.pi * x[1]))**6
    g = 2 - temp1
    
    y[0] = x[0]
    y[1] = g / x[0]
    
    return y

def MMF10(x):
    # 0.1 < x1 <= 1.1, 0.1 <= x2 <= 1.1, g(x2) > 0
    
    y = np.zeros(2)
    
    g = 2 - np.exp(-((x[1] - 0.2) / 0.004)**2) - 0.8 * np.exp(-((x[1] - 0.6) / 0.4)**2)
    
    y[0] = x[0]
    y[1] = g / x[0]
    
    return y

def MMF10_l(x):
    # 0.1 < x1 <= 1.1, 0.1 <= x2 <= 1.1, g(x2) > 0
    
    y = np.zeros(2)
    
    g = 2 - np.exp(-((x[1] - 0.2) / 0.004)**2) - 0.8 * np.exp(-((x[1] - 0.6) / 0.4)**2)
    
    y[0] = x[0]
    y[1] = g / x[0]
    
    return y

def MMF11(x):
    # 0.1 <= x1 <= 1.1, 0.1 <= x2 <= 1.1
    
    y = np.zeros(2)
    num_of_peak = 2
    x1 = x[0]
    x2 = x[1]
    
    temp1 = (np.sin(num_of_peak * np.pi * x2))**6
    temp2 = np.exp(-2 * np.log10(2) * ((x2 - 0.1) / 0.8)**2)
    g = 2 - temp2 * temp1
    
    y[0] = x1
    y[1] = g / x1
    
    return y

def MMF11_l(x):
    # 0.1 <= x1 <= 1.1, 0.1 <= x2 <= 1.1
    
    y = np.zeros(2)
    num_of_peak = 2
    x1 = x[0]
    x2 = x[1]
    
    temp1 = (np.sin(num_of_peak * np.pi * x2))**6
    temp2 = np.exp(-2 * np.log10(2) * ((x2 - 0.1) / 0.8)**2)
    g = 2 - temp2 * temp1
    
    y[0] = x1
    y[1] = g / x1
    
    return y

def MMF12(x):
    # 0 <= x1 <= 1, 0 <= x2 <= 1
    q = 4
    Alfa = 2

    y = np.zeros(2)
    y[0] = x[0]

    num_of_peak = 2
    g = 2 - ((np.sin(num_of_peak * np.pi * x[1]))**6) * (np.exp(-2 * np.log10(2) * ((x[1] - 0.1) / 0.8)**2))
    h = 1 - (y[0] / g)**Alfa - (y[0] / g) * np.sin(2 * np.pi * q * y[0])
    
    y[1] = g * h
    
    return y

def MMF12_l(x):
    # 0 <= x1 <= 1, 0 <= x2 <= 1
    q = 4
    Alfa = 2

    y = np.zeros(2)
    y[0] = x[0]

    num_of_peak = 2
    g = 2 - ((np.sin(num_of_peak * np.pi * x[1]))**6) * (np.exp(-2 * np.log10(2) * ((x[1] - 0.1) / 0.8)**2))
    h = 1 - (y[0] / g)**Alfa - (y[0] / g) * np.sin(2 * np.pi * q * y[0])
    
    y[1] = g * h
    
    return y



def MMF13(x):
    # 0.1 <= x1 <= 1.1, 0.1 <= x2 <= 1.1, 0.1 <= x3 <= 1.1
    
    y = np.zeros(2)
    g = 2 - np.exp(-2 * np.log10(2) * ((x[1] + np.sqrt(x[2]) - 0.1) / 0.8)**2) * (np.sin(2 * np.pi * (x[1] + np.sqrt(x[2])))**6)
    y[0] = x[0]
    y[1] = g / x[0]

    return y

def MMF13_l(x):
    # 0.1 <= x1 <= 1.1, 0.1 <= x2 <= 1.1, 0.1 <= x3 <= 1.1
    
    y = np.zeros(2)
    g = 2 - np.exp(-2 * np.log10(2) * ((x[1] + np.sqrt(x[2]) - 0.1) / 0.8)**2) * (np.sin(2 * np.pi * (x[1] + np.sqrt(x[2])))**6)
    y[0] = x[0]
    y[1] = g / x[0]

    return y

def MMF14(x, M=3, num_of_peak=2):
    # 0 <= xi <= 1
    x=x.reshape(1, 3)
    g = 2 - (np.sin(num_of_peak * np.pi * x[:, -1]))**2

    y = (1 + g) * np.fliplr(np.cumprod(np.hstack((np.ones((x.shape[0], 1)), np.cos(x[:, :M-1] * np.pi / 2))), axis=1)) * np.hstack((np.ones((x.shape[0], 1)), np.sin(x[:, M-2::-1] * np.pi / 2)))
    y = y.T
    y=y.reshape(-1)
    return y

def MMF14_a(x, M=3, num_of_peak=2):
    # 0 <= xi <= 1
    x=x.reshape(1, 3)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    x_g = x[:, -1] - 0.5 * np.sin(np.pi * x[:, -2])

    g = 2 - (np.sin(num_of_peak * np.pi * (x_g + 1 / (2 * num_of_peak))))**2

    y = (1 + g) * np.fliplr(np.cumprod(np.hstack((np.ones((x.shape[0], 1)), np.cos(x[:, :M-1] * np.pi / 2))), axis=1)) * np.hstack((np.ones((x.shape[0], 1)), np.sin(x[:, M-2::-1] * np.pi / 2)))
    y = y.T
    y=y.reshape(-1)
    return y

def MMF15(x, M=3, num_of_peak=2):
    # 0 <= xi <= 1
    x=x.reshape(1, 3)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    
    g = 2 - np.exp(-2 * np.log10(2) * ((x[:, -1] - 0.1) / 0.8)**2) * (np.sin(num_of_peak * np.pi * x[:, -1]))**2

    y = (1 + g) * np.fliplr(np.cumprod(np.hstack((np.ones((x.shape[0], 1)), np.cos(x[:, :M-1] * np.pi / 2))), axis=1)) * np.hstack((np.ones((x.shape[0], 1)), np.sin(x[:, M-2::-1] * np.pi / 2)))
    y = y.T
    y=y.reshape(-1)
    return y

def MMF15_l(x, M=3, num_of_peak=2):
    # 0 <= xi <= 1
    x=x.reshape(1, 3)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    
    g = 2 - np.exp(-2 * np.log10(2) * ((x[:, -1] - 0.1) / 0.8)**2) * (np.sin(num_of_peak * np.pi * x[:, -1]))**2

    y = (1 + g) * np.fliplr(np.cumprod(np.hstack((np.ones((x.shape[0], 1)), np.cos(x[:, :M-1] * np.pi / 2))), axis=1)) * np.hstack((np.ones((x.shape[0], 1)), np.sin(x[:, M-2::-1] * np.pi / 2)))
    y = y.T
    y=y.reshape(-1)
    return y

def MMF15_a(x, M=3, num_of_peak=2):
    # 0 <= xi <= 1
    x=x.reshape(1, 3)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    
    t = (-0.5 * np.sin(np.pi * x[:, -2]) + x[:, -1])
    g = 2 - np.exp(-2 * np.log10(2) * ((t + 1 / (2 * num_of_peak) - 0.1) / 0.8)**2) * (np.sin(num_of_peak * np.pi * (t + 1 / (2 * num_of_peak))))**2

    y = (1 + g) * np.fliplr(np.cumprod(np.hstack((np.ones((x.shape[0], 1)), np.cos(x[:, :M-1] * np.pi / 2))), axis=1)) * np.hstack((np.ones((x.shape[0], 1)), np.sin(x[:, M-2::-1] * np.pi / 2)))
    y = y.T
    y=y.reshape(-1)
    return y

def MMF15_a_l(x, M=3, num_of_peak=2):
    # 0 <= xi <= 1
    x=x.reshape(1, 3)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    
    t = (-0.5 * np.sin(np.pi * x[:, -2]) + x[:, -1])
    g = 2 - np.exp(-2 * np.log10(2) * ((t + 1 / (2 * num_of_peak) - 0.1) / 0.8)**2) * (np.sin(num_of_peak * np.pi * (t + 1 / (2 * num_of_peak))))**2

    y = (1 + g) * np.fliplr(np.cumprod(np.hstack((np.ones((x.shape[0], 1)), np.cos(x[:, :M-1] * np.pi / 2))), axis=1)) * np.hstack((np.ones((x.shape[0], 1)), np.sin(x[:, M-2::-1] * np.pi / 2)))
    y = y.T
    y=y.reshape(-1)
    return y

def MMF16_l1(x, M=3):
    x=x.reshape(1, 3)
    num_of_g_peak = 2  # number of global PSs
    num_of_l_peak = 1  # number of local PSs
    g = np.zeros((x.shape[0], M))
    
    for i in range(x.shape[0]):
        if 0 <= x[i, -1] < 0.5:
            g[i, :] = 2 - (np.sin(2 * num_of_g_peak * np.pi * x[i, -1])) ** 2
        elif 0.5 <= x[i, -1] <= 1:
            g[i, :] = 2 - np.exp(-2 * np.log10(2) * ((x[i, -1] - 0.1) / 0.8) ** 2) * \
                      (np.sin(2 * num_of_l_peak * np.pi * x[i, -1])) ** 2

    cos_vals = np.cos(x[:, :M-1] * np.pi / 2)
    sin_vals = np.sin(x[:, M-2::-1] * np.pi / 2)
    y = (1 + g) * np.fliplr(np.cumprod(np.hstack((np.ones((g.shape[0], 1)), cos_vals)), axis=1)) * \
        np.hstack((np.ones((g.shape[0], 1)), sin_vals))
    
    return y

def MMF16_l2(x, M=3):
    x=x.reshape(1, 3)
    num_of_g_peak = 1  # number of global PSs
    num_of_l_peak = 2  # number of local PSs
    g = np.zeros((x.shape[0], M))
    
    for i in range(x.shape[0]):
        if 0 <= x[i, -1] < 0.5:
            g[i, :] = 2 - (np.sin(2 * num_of_g_peak * np.pi * x[i, -1])) ** 2
        elif 0.5 <= x[i, -1] <= 1:
            g[i, :] = 2 - np.exp(-2 * np.log10(2) * ((x[i, -1] - 0.1) / 0.8) ** 2) * \
                      (np.sin(2 * num_of_l_peak * np.pi * x[i, -1])) ** 2

    cos_vals = np.cos(x[:, :M-1] * np.pi / 2)
    sin_vals = np.sin(x[:, M-2::-1] * np.pi / 2)
    y = (1 + g) * np.fliplr(np.cumprod(np.hstack((np.ones((g.shape[0], 1)), cos_vals)), axis=1)) * \
        np.hstack((np.ones((g.shape[0], 1)), sin_vals))
    
    return y

def MMF16_l3(x, M=3):
    x=x.reshape(1, 3)
    num_of_g_peak = 2  # number of global PSs
    num_of_l_peak = 2  # number of local PSs
    g = np.zeros((x.shape[0], M))
    
    for i in range(x.shape[0]):
        if 0 <= x[i, -1] < 0.5:
            g[i, :] = 2 - (np.sin(2 * num_of_g_peak * np.pi * x[i, -1])) ** 2
        elif 0.5 <= x[i, -1] <= 1:
            g[i, :] = 2 - np.exp(-2 * np.log10(2) * ((x[i, -1] - 0.1) / 0.8) ** 2) * \
                      (np.sin(2 * num_of_l_peak * np.pi * x[i, -1])) ** 2

    cos_vals = np.cos(x[:, :M-1] * np.pi / 2)
    sin_vals = np.sin(x[:, M-2::-1] * np.pi / 2)
    y = (1 + g) * np.fliplr(np.cumprod(np.hstack((np.ones((g.shape[0], 1)), cos_vals)), axis=1)) * \
        np.hstack((np.ones((g.shape[0], 1)), sin_vals))
    
    return y

def Omni_test(x):
    f = np.zeros(2)
    n = len(x)
    
    for i in range(n):
        f[0] += np.sin(np.pi * x[i])
        f[1] += np.cos(np.pi * x[i])

    return f

def fun(x, a):
    y = np.zeros(2)
    y[0] = (x[0] + a)**2 + x[1]**2
    y[1] = (x[0] - a)**2 + x[1]**2
    
    return y   

def SYM_PART_rotated(x):
    a = 1
    b = 10
    c = 8
    w = np.pi / 4
    #print("x",x)
    xx1 = np.cos(w) * x[0] - np.sin(w) * x[1]
    xx2 = np.sin(w) * x[0] + np.cos(w) * x[1]
    #print("xx1",xx1)
    #print("xx2",xx2)
    x[0] = xx1
    x[1] = xx2
    #print("x",x)

    temp_t1 = np.sign(x[0]) * np.ceil((np.abs(x[0]) - (a + c/2)) / (2*a + c))
    temp_t2 = np.sign(x[1]) * np.ceil((np.abs(x[1]) - b/2) / b)
    t1 = np.sign(temp_t1) * min(np.abs(temp_t1), 1)
    t2 = np.sign(temp_t2) * min(np.abs(temp_t2), 1)
    
    #print("x",x)
    x1 = x[0] - t1 * (c + 2 * a)
    x2 = x[1] - t2 * b
    #print("x1",x1)
    #print("x2",x2)
    return fun([x1, x2], a)


def SYM_PART_simple(x):
    a = 1
    b = 10
    c = 8
    
    temp_t1 = np.sign(x[0]) * np.ceil((np.abs(x[0]) - (a + c/2)) / (2*a + c))
    temp_t2 = np.sign(x[1]) * np.ceil((np.abs(x[1]) - b/2) / b)
    t1 = np.sign(temp_t1) * min(np.abs(temp_t1), 1)
    t2 = np.sign(temp_t2) * min(np.abs(temp_t2), 1)
    
    x1 = x[0] - t1 * (c + 2 * a)
    x2 = x[1] - t2 * b
    
    return fun([x1, x2], a)

#IDMP数据集
def IDMPM2T1(x):
    # 双目标两变量测试函数，mu=3,type1
    mu = 3
    f = np.zeros(2)
    g1 = abs(x[1] + 0.5)
    g2 = mu * abs(x[1] - 0.5)
    
    f[0] = min(abs(x[0] + 0.6) + g1, abs(x[0] - 0.4) + g2)
    f[1] = min(abs(x[0] + 0.4) + g1, abs(x[0] - 0.6) + g2)
    
    return f

def IDMPM2T2(x):
    x=x.reshape(1, 2)
    M = 2  # 目标数
    NP = 2
    psize = 0.10 * np.ones(NP)
    center = np.array([-0.50, 0.50])
    
    Points = np.vstack([center - psize, center + psize])
    
    PopDec = x
    N = len(PopDec)
    PopObj = np.empty((N, M))
    a = 0.4
    
    for i in range(M):
        temp = np.abs(np.tile(PopDec[:, 0], (1, NP)) - np.tile(Points[i, :], (N, 1)))
        temp[:, 0] += 100 * np.power(np.abs(PopDec[:, 1] + 0.5), 2)
        temp[:, 1] += 100 * np.power(np.abs(PopDec[:, 1] - 0.5), (2 - a))
        PopObj[:, i] = np.min(temp, axis=1)
    
    return PopObj

def IDMPM2T3(x):
    x=x.reshape(1, 2)
    M = 2  # 目标数
    NP = 2
    psize = 0.10 * np.ones(NP)
    center = np.array([-0.50, 0.50])
    
    Points = np.vstack([center - psize, center + psize])
    
    PopDec = x
    N = len(PopDec)
    PopObj = np.empty((N, M))
    a = 0.4
    
    for i in range(M):
        temp = np.abs(np.tile(PopDec[:, 0], (1, NP)) - np.tile(Points[i, :], (N, 1)))
        temp[:, 0] += 100 * np.power(np.abs(PopDec[:, 1] + 0.5), 2)
        temp[:, 1] += 100 * np.power(PopDec[:, 1] - 0.5 + a * (PopDec[:, 0] - 0.5), 2)
        PopObj[:, i] = np.min(temp, axis=1)
    
    return PopObj

def IDMPM2T4(x):
    x=x.reshape(1, 2)
    M = 2  # 目标数
    NP = 2
    psize = 0.10 * np.ones(NP)
    center = np.array([-0.50, 0.50])
    
    Points = np.vstack([center - psize, center + psize])
    
    PopDec = x
    N = len(PopDec)
    PopObj = np.empty((N, M))
    a = 4
    
    for i in range(M):
        temp = np.abs(np.tile(PopDec[:, 0], (1, NP)) - np.tile(Points[i, :], (N, 1)))
        temp[:, 0] += 100 * (np.power(np.abs(PopDec[:, 1] + 0.50), 2) + 1 - np.cos(1 * 2 * pi * (PopDec[:, 1] + 0.50)))
        temp[:, 1] += 100 * (np.power(np.abs(PopDec[:, 1] - 0.50), 2) + 1 - np.cos(a * 2 * pi * (PopDec[:, 1] - 0.50)))
        PopObj[:, i] = np.min(temp, axis=1)
    
    return PopObj

def IDMPM3T1(x):
    M = 3  # 目标数
    x=x.reshape(1,3)
    pgon = Polygon([(-0.866025403784439, -0.500000000000000), (0, 1), (0.866025403784439, -0.500000000000000)])
    psize = [0.1, 0.1, 0.1, 0.1]
    center = [[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]]
    
    Points = np.zeros((M, 2, 4))
    for i in range(4):
        #print("np.array(pgon.exterior.coords)",(np.array(pgon.exterior.coords) ).shape)
        Points[:,:,i] = np.array(pgon.exterior.coords[:3]) * psize[i] + center[i]
    
    N = x.shape[0]
    PopObj = np.zeros((N, M))
    for i in range(M):
        temp = cdist(x[:, :2], np.reshape(Points[i,:,:], (2,4)).T)
        temp[:,0] += 1 * np.abs(x[:,2] + 0.6)
        temp[:,1] += 2 * np.abs(x[:,2] + 0.2)
        temp[:,2] += 3 * np.abs(x[:,2] - 0.2)
        temp[:,3] += 4 * np.abs(x[:,2] - 0.6)
        PopObj[:,i] = np.min(temp, axis=1)
    
    return PopObj

def IDMPM3T2(x):
    M = 3  # 目标数
    x=x.reshape(1,3)
    pgon = Polygon([(-0.866025403784439, -0.500000000000000), (0, 1), (0.866025403784439, -0.500000000000000)])
    psize = np.array([0.1, 0.1, 0.1, 0.1])
    center = np.array([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])

    Points = np.zeros((M, 2, 4))
    for i in range(4):
        Points[:,:,i] = np.array(pgon.exterior.coords[:3]) * psize[i] + center[i]

    N = x.shape[0]
    PopObj = np.empty((N, M))

    for i in range(M):
        temp = cdist(x[:,:2], np.transpose(Points[i,:,:]))
        temp[:,0] += 100 * np.power(np.abs(x[:,2] + 0.6), 2)
        temp[:,1] += 100 * np.power(np.abs(x[:,2] + 0.2), 1.8)
        temp[:,2] += 100 * np.power(np.abs(x[:,2] - 0.2), 1.6)
        temp[:,3] += 100 * np.power(np.abs(x[:,2] - 0.6), 1.4)
        PopObj[:,i] = np.min(temp, axis=1)

    return PopObj

def IDMPM3T3(x):
    M = 3  # 目标数
    x=x.reshape(1,3)
    pgon = Polygon([(-0.866025403784439, -0.500000000000000), (0, 1), (0.866025403784439, -0.500000000000000)])
    psize = np.array([0.1, 0.1, 0.1, 0.1])
    center = np.array([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])

    Points = np.zeros((M, 2, 4))
    for i in range(4):
        Points[:,:,i] =  np.array(pgon.exterior.coords[:3]) * psize[i] + center[i]

    N = x.shape[0]
    PopObj = np.empty((N, M))

    for i in range(M):
        temp = cdist(x[:,:2], np.transpose(Points[i,:,:]))
        temp[:,0] += 100 * np.power(x[:,2] + 0.6, 2)
        t2 = x[:,0] - 0.5 + x[:,1] + 0.5
        temp[:,1] += 100 * np.power(x[:,2] + 0.2 + 0.1 * t2, 2)
        t3 = x[:,0] - 0.5 + x[:,1] - 0.5
        temp[:,2] += 100 * np.power(x[:,2] - 0.2 + 0.2 * t3, 2)
        t4 = x[:,0] + 0.5 + x[:,1] - 0.5
        temp[:,3] += 100 * np.power(x[:,2] - 0.6 + 0.3 * t4, 2)
        PopObj[:,i] = np.min(temp, axis=1)
    
    return PopObj

def IDMPM3T4(x):
    M = 3  # 目标数
    x=x.reshape(1,3)
    pgon = Polygon([(-0.866025403784439, -0.500000000000000), (0, 1), (0.866025403784439, -0.500000000000000)])
    psize = np.array([0.1, 0.1, 0.1, 0.1])
    center = np.array([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])

    Points = np.zeros((M, 2, 4))
    for i in range(4):
        Points[:,:,i] = np.array(pgon.exterior.coords[:3]) * psize[i] + center[i]

    N = x.shape[0]
    PopObj = np.empty((N, M))

    for i in range(M):
        temp = cdist(x[:, :2], np.reshape(Points[i,:,:], (2, 4)).T)
        temp[:,0] += 100 * ((x[:,2] + 0.6)**2 + 1 - np.cos(1 * 2 * np.pi * (x[:,2] + 0.6)))
        temp[:,1] += 100 * ((x[:,2] + 0.2)**2 + 1 - np.cos(2 * 2 * np.pi * (x[:,2] + 0.2)))
        temp[:,2] += 100 * ((x[:,2] - 0.2)**2 + 1 - np.cos(3 * 2 * np.pi * (x[:,2] - 0.2)))
        temp[:,3] += 100 * ((x[:,2] - 0.6)**2 + 1 - np.cos(4 * 2 * np.pi * (x[:,2] - 0.6)))
        PopObj[:,i] = np.min(temp, axis=1)
    
    return PopObj

def IDMPM4T1(x):
    M = 4  # 目标数
    x=x.reshape(1,4)
    pgon = Polygon([[-0.707106781186548, -0.707106781186548], [-0.707106781186548, 0.707106781186548], [0.707106781186548, 0.707106781186548], [0.707106781186548, -0.707106781186548]])
    psize = np.array([0.1, 0.1, 0.1, 0.1])
    center = np.array([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])

    Points = np.zeros((M, 2, 4))
    for i in range(4):
        Points[:,:,i] = np.array(pgon.exterior.coords[:4]) * psize[i] + center[i]

    N = x.shape[0]
    PopDec = x
    PopObj = np.empty((N, M))

    for i in range(M):
        temp = cdist(PopDec[:, :2], np.reshape(Points[i,:,:], (2, 4)).T)
        temp[:,0] += 1 * (np.abs(PopDec[:,2] + 0.6) + np.abs(PopDec[:,3] + 0.6))
        temp[:,1] += 2 * (np.abs(PopDec[:,2] + 0.2) + np.abs(PopDec[:,3] + 0.2))
        temp[:,2] += 3 * (np.abs(PopDec[:,2] - 0.2) + np.abs(PopDec[:,3] - 0.2))
        temp[:,3] += 4 * (np.abs(PopDec[:,2] - 0.6) + np.abs(PopDec[:,3] - 0.6))
        PopObj[:,i] = np.min(temp, axis=1)
    
    return PopObj

def IDMPM4T2(x):
    M = 4  # 目标数
    x=x.reshape(1,4)
    pgon = Polygon([[-0.707106781186548, -0.707106781186548], [-0.707106781186548, 0.707106781186548], [0.707106781186548, 0.707106781186548], [0.707106781186548, -0.707106781186548]])
    psize = np.array([0.1, 0.1, 0.1, 0.1])
    center = np.array([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])

    Points = np.zeros((M, 2, 4))
    for i in range(4):
        Points[:,:,i] = np.array(pgon.exterior.coords[:4]) * psize[i] + center[i]

    N = x.shape[0]
    PopDec = x
    PopObj = np.empty((N, M))

    for i in range(M):
        temp = cdist(PopDec[:, :2], np.reshape(Points[i,:,:], (2, 4)).T)
        temp[:,0] += 100 * ((np.abs(PopDec[:,2] + 0.6))**2 + (np.abs(PopDec[:,3] + 0.6))**2)
        temp[:,1] += 100 * ((np.abs(PopDec[:,2] + 0.2))**1.8 + (np.abs(PopDec[:,3] + 0.2))**1.8)
        temp[:,2] += 100 * ((np.abs(PopDec[:,2] - 0.2))**1.6 + (np.abs(PopDec[:,3] - 0.2))**1.6)
        temp[:,3] += 100 * ((np.abs(PopDec[:,2] - 0.6))**1.4 + (np.abs(PopDec[:,3] - 0.6))**1.4)
        PopObj[:,i] = np.min(temp, axis=1)
    
    return PopObj

def IDMPM4T3(x):
    M = 4  # 目标数
    x=x.reshape(1,4)
    pgon = Polygon([[-0.707106781186548, -0.707106781186548], [-0.707106781186548, 0.707106781186548], [0.707106781186548, 0.707106781186548], [0.707106781186548, -0.707106781186548]])
    psize = np.array([0.1, 0.1, 0.1, 0.1])
    center = np.array([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])
    Points = np.empty((M, 2, 4))
    
    for i in range(4):
        Points[:,:,i] = np.array(pgon.exterior.coords[:4]) * psize[i] + center[i]
        
    N = x.shape[0]
    PopDec = x
    PopObj = np.empty((N, M))
    
    for i in range(M):
        temp = cdist(PopDec[:, :2], Points[i].T)
        temp[:,0] += 100 * ((PopDec[:,2] + 0.6)**2 + (PopDec[:,3] + 0.6)**2)
        t2 = PopDec[:,0] - 0.5 + PopDec[:,1] + 0.5
        a2 = 0.05
        temp[:,1] += 100 * (PopDec[:,2] + 0.2 + a2 * t2)**2 + 100 * (PopDec[:,3] + 0.2 + a2 * t2)**2
        a3 = 0.1
        t3 = PopDec[:,0] - 0.5 + PopDec[:,1] - 0.5
        temp[:,2] += 100 * (PopDec[:,2] - 0.2 + a3 * t3)**2 + 100 * (PopDec[:,3] - 0.2 + a3 * t3)**2
        a4 = 0.15
        t4 = PopDec[:,0] + 0.5 + PopDec[:,1] - 0.5
        temp[:,3] += 100 * (PopDec[:,2] - 0.6 + a4 * t4)**2 + 100 * (PopDec[:,3] - 0.6 + a4 * t4)**2
        PopObj[:,i] = np.min(temp, axis=1)
    
    return PopObj

def IDMPM4T4(x):
    M = 4  # 目标数
    x=x.reshape(1,4)
    pgon = Polygon([[-0.707106781186548, -0.707106781186548], [-0.707106781186548, 0.707106781186548], [0.707106781186548, 0.707106781186548], [0.707106781186548, -0.707106781186548]])
    psize = np.array([0.1, 0.1, 0.1, 0.1])
    center = np.array([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])
    Points = np.empty((M, 2, 4))
    
    for i in range(4):
        Points[:,:,i] = np.array(pgon.exterior.coords[:4]) * psize[i] + center[i]
        
    N = x.shape[0]
    PopDec = x
    PopObj = np.empty((N, M))
    
    for i in range(M):
        temp = cdist(PopDec[:, :2], Points[i].T)
        temp[:,0] += 100 * ((PopDec[:,2] + 0.6)**2 + 1 - np.cos(1 * 2 * np.pi * (PopDec[:,2] + 0.6))) + 100 * ((PopDec[:,3] + 0.6)**2 + 1 - np.cos(0 * 2 * np.pi * (PopDec[:,3] + 0.6)))
        temp[:,1] += 100 * ((PopDec[:,2] + 0.2)**2 + 1 - np.cos(2 * 2 * np.pi * (PopDec[:,2] + 0.2))) + 100 * ((PopDec[:,3] + 0.2)**2 + 1 - np.cos(0 * 2 * np.pi * (PopDec[:,3] + 0.2)))
        temp[:,2] += 100 * ((PopDec[:,2] - 0.2)**2 + 1 - np.cos(3 * 2 * np.pi * (PopDec[:,2] - 0.2))) + 100 * ((PopDec[:,3] - 0.2)**2 + 1 - np.cos(0 * 2 * np.pi * (PopDec[:,3] - 0.2)))
        temp[:,3] += 100 * ((PopDec[:,2] - 0.6)**2 + 1 - np.cos(4 * 2 * np.pi * (PopDec[:,2] - 0.6))) + 100 * ((PopDec[:,3] - 0.6)**2 + 1 - np.cos(0 * 2 * np.pi * (PopDec[:,3] - 0.6)))
        PopObj[:,i] = np.min(temp, axis=1)
    
    return PopObj

#MMMOP数据集：
def MMMOP1A(PopDec):
    PopDec=PopDec.reshape(1,3)
    kA = 1
    kB = 1

    N, _ = PopDec.shape
    M = 2
    #a=kA + kB - np.sum(np.sin(5 * np.pi * PopDec[:, M-1:M + kA - 2]) ** 6, axis=1)
    #b=np.sum((PopDec[:, M + kA-1:] - 0.5) ** 2 - np.cos(20 * np.pi * (PopDec[:, M + kA-1:] - 0.5)), axis=1)
    g = 100 * (kA + kB - np.sum(np.sin(5 * np.pi * PopDec[:, M-1:M + kA - 1]) ** 6, axis=1) + np.sum((PopDec[:, M + kA-1:] - 0.5) ** 2 - np.cos(20 * np.pi * (PopDec[:, M + kA-1:] - 0.5)), axis=1))
    #print("g:",g)
    
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack([np.ones((N, 1)), PopDec[:, :M-1]]), axis=1)) * np.hstack([np.ones((N, 1)), 1 - PopDec[:, M-2::-1]])

    return PopObj

def MMMOP1B(PopDec):
    PopDec=PopDec.reshape(1,7)
    kA = 1
    kB = 4

    N, _ = PopDec.shape
    M = 3
    #a=kA + kB - np.sum(np.sin(5 * np.pi * PopDec[:, M-1:M + kA - 2]) ** 6, axis=1)
    #b=np.sum((PopDec[:, M + kA-1:] - 0.5) ** 2 - np.cos(20 * np.pi * (PopDec[:, M + kA-1:] - 0.5)), axis=1)
    g = 100 * (kA + kB - np.sum(np.sin(5 * np.pi * PopDec[:, M-1:M + kA - 1]) ** 6, axis=1) + np.sum((PopDec[:, M + kA-1:] - 0.5) ** 2 - np.cos(20 * np.pi * (PopDec[:, M + kA-1:] - 0.5)), axis=1))
    #print("g:",g)
    
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack([np.ones((N, 1)), PopDec[:, :M-1]]), axis=1)) * np.hstack([np.ones((N, 1)), 1 - PopDec[:, M-2::-1]])

    return PopObj

def MMMOP2A(PopDec):
    PopDec=PopDec.reshape(1,3)
    kA = 1
    kB = 1
    alpha = 100


    M = 2

    PopDec[:, :M-1] = np.power(PopDec[:, :M-1], alpha)
    y = 9.75 * PopDec[:, M-1:M+kA-1] + 0.25
    #print("y",y)
    g = kA - np.sum(np.sin(10 * np.log(y)), axis=1) + np.sum((PopDec[:, M-1+kA:] - 0.5) ** 2, axis=1)
    #print("g",g)
    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(PopDec[:, :M-1] * np.pi/2))), axis=1)) * \
             np.hstack((ones_mat, np.sin(PopDec[:, M-2::-1] * np.pi/2)))

    return PopObj

def MMMOP2B(PopDec):
    PopDec=PopDec.reshape(1,7)
    kA = 1
    kB = 4
    alpha = 100

    M = 3

    PopDec[:, :M-1] = np.power(PopDec[:, :M-1], alpha)
    y = 9.75 * PopDec[:, M-1:M+kA-1] + 0.25
    #print("y",y)
    g = kA - np.sum(np.sin(10 * np.log(y)), axis=1) + np.sum((PopDec[:, M-1+kA:] - 0.5) ** 2, axis=1)
    #print("g",g)
    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(PopDec[:, :M-1] * np.pi/2))), axis=1)) * \
             np.hstack((ones_mat, np.sin(PopDec[:, M-2::-1] * np.pi/2)))

    return PopObj

def MMMOP3A(PopDec):
    PopDec=PopDec.reshape(1,2)
    kA = 0
    kB = 1
    c = 0
    d = 3
    
    M = 2

    y = PopDec[:, :M-1] * d - np.floor(PopDec[:, :M-1] * d)
    g = kA + np.sum((PopDec[:, M-1+kA:] - 0.5) ** 2, axis=1)
    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(y * np.pi/2))), axis=1)) * \
             np.hstack((ones_mat, np.sin(y[:, M-2::-1] * np.pi/2)))

    return PopObj

def MMMOP3B(PopDec):
    PopDec=PopDec.reshape(1,7)
    kA = 0
    kB = 5
    c = 0
    d = 1

    M = 3

    y = PopDec[:, :M-1] * d - np.floor(PopDec[:, :M-1] * d)
    g = kA + np.sum((PopDec[:, M-1+kA:] - 0.5) ** 2, axis=1)
    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(y * np.pi/2))), axis=1)) * \
             np.hstack((ones_mat, np.sin(y[:, M-2::-1] * np.pi/2)))
    return PopObj

def MMMOP3C(PopDec):
    PopDec=PopDec.reshape(1,6)
    kA = 1
    kB = 4
    c = 3
    d = 3
    
    M = 2

    y = PopDec[:, :M-1] * d - np.floor(PopDec[:, :M-1] * d)
    
    g = kA + np.sum(np.cos(2 * np.pi * c * PopDec[:, M-1:M-1+kA]), axis=1) + \
        np.sum((PopDec[:, M-1+kA:] - 0.5) ** 2, axis=1)
    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(y * np.pi/2))), axis=1)) * \
             np.hstack((ones_mat, np.sin(y[:, M-2::-1] * np.pi/2)))

    return PopObj

def MMMOP3D(PopDec):
    PopDec=PopDec.reshape(1,7)
    kA = 1
    kB = 4
    c = 2
    d = 2

    M = 3

    y = PopDec[:, :M-1] * d - np.floor(PopDec[:, :M-1] * d)
    g = kA + np.sum(np.cos(2 * np.pi * c * PopDec[:, M-1:M-1+kA]), axis=1) + \
        np.sum((PopDec[:, M-1+kA:] - 0.5) ** 2, axis=1)
    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(y * np.pi/2))), axis=1)) * \
             np.hstack((ones_mat, np.sin(y[:, M-2::-1] * np.pi/2)))

    return PopObj

def MMMOP4A(PopDec):
    PopDec = PopDec.reshape(1,2)

    kA = 0
    kB = 1
    c = 0
    d = 4

    
    M = 2
    N,_= PopDec.shape

    y = np.zeros((N, M-1))
    for i in range(N):
        for j in range(M-1):
            dsum = 0
            y[i, j] = PopDec[i, j] * d * (d+1) * 0.5
            for m in range(d, 0, -1):
                dsum = dsum + m
                if y[i, j] <= dsum:
                    y[i, j] = (y[i, j] - dsum + m) / d
                    break

    g = 100 * (kA + kB + np.sum((PopDec[:, M-1+kA:] - 0.5) ** 2 - np.cos(20 * np.pi * (PopDec[:, M-1+kA:] - 0.5)), axis=1))
    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(y * np.pi/2))), axis=1)) * \
             np.hstack((ones_mat, np.sin(y[:, M-2::-1] * np.pi/2)))

    return PopObj

def MMMOP4B(PopDec):
    PopDec = PopDec.reshape(1,7)

    kA = 0
    kB = 5
    c = 0
    d = 3

    
    M = 3
    N,_= PopDec.shape

    y = np.zeros((N, M-1))
    for i in range(N):
        for j in range(M-1):
            dsum = 0
            y[i, j] = PopDec[i, j] * d * (d+1) * 0.5
            for m in range(d, 0, -1):
                dsum = dsum + m
                if y[i, j] <= dsum:
                    y[i, j] = (y[i, j] - dsum + m) / d
                    break

    g = 100 * (kA + kB + np.sum((PopDec[:, M-1+kA:] - 0.5) ** 2 - np.cos(20 * np.pi * (PopDec[:, M-1+kA:] - 0.5)), axis=1))
    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(y * np.pi/2))), axis=1)) * \
             np.hstack((ones_mat, np.sin(y[:, M-2::-1] * np.pi/2)))

    return PopObj

def MMMOP4C(PopDec):
    PopDec = PopDec.reshape(1,6)

    kA = 1
    kB = 4
    c = 2
    d = 4
    
    M = 2
    N,_= PopDec.shape

    y = np.zeros((N, M-1))
    for i in range(N):
        for j in range(M-1):
            dsum = 0
            y[i, j] = PopDec[i, j] * d * (d+1) * 0.5
            for m in range(d, 0, -1):
                dsum = dsum + m
                if y[i, j] <= dsum:
                    y[i, j] = (y[i, j] - dsum + m) / d
                    break

    g = 100 * (kA + kB + np.sum(np.cos(2 * np.pi * c * PopDec[:, M-1:M-1+kA]), axis=1) +
               np.sum((PopDec[:, M-1+kA:] - 0.5) ** 2 - np.cos(20 * np.pi * (PopDec[:, M-1+kA:] - 0.5)), axis=1))
    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(y * np.pi/2))), axis=1)) * \
             np.hstack((ones_mat, np.sin(y[:, M-2::-1] * np.pi/2)))

    return PopObj

def MMMOP4D(PopDec):
    PopDec = PopDec.reshape(1,7)

    kA = 1
    kB = 4
    c = 2
    d = 3

    M = 3
    N,_= PopDec.shape

    y = np.zeros((N, M-1))
    for i in range(N):
        for j in range(M-1):
            dsum = 0
            y[i, j] = PopDec[i, j] * d * (d+1) * 0.5
            for m in range(d, 0, -1):
                dsum = dsum + m
                if y[i, j] <= dsum:
                    y[i, j] = (y[i, j] - dsum + m) / d
                    break

    g = 100 * (kA + kB + np.sum(np.cos(2 * np.pi * c * PopDec[:, M-1:M-1+kA]), axis=1) +
               np.sum((PopDec[:, M-1+kA:] - 0.5) ** 2 - np.cos(20 * np.pi * (PopDec[:, M-1+kA:] - 0.5)), axis=1))
    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(y * np.pi/2))), axis=1)) * \
             np.hstack((ones_mat, np.sin(y[:, M-2::-1] * np.pi/2)))

    return PopObj


def MMMOP5A(PopDec):
    PopDec = PopDec.reshape(1,2)
    kA = 0
    kB = 1
    c = 0
    d = 3

    
    M = 2
    N,_= PopDec.shape

    y = np.zeros((N, M-1))
    for i in range(N):
        for j in range(M-1):
            dsum = 0
            y[i, j] = PopDec[i, j] * (2**(d+1) - 1)
            for m in range(d, -1, -1):
                dsum = dsum + 2**m
                if y[i, j] <= dsum:
                    y[i, j] = (y[i, j] - dsum + 2**m) / 2**m
                    break

    g = 100 * (kA + kB + np.sum((PopDec[:, M-1+kA:] - 0.5)**2 - np.cos(20 * np.pi * (PopDec[:, M-1+kA:] - 0.5)), axis=1))

    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(y * np.pi/2))), axis=1)) * \
             np.hstack((ones_mat, np.sin(y[:, M-2::-1] * np.pi/2)))

    return PopObj

def MMMOP5B(PopDec):
    PopDec = PopDec.reshape(1,7)
    kA = 0
    kB = 5
    c = 0
    d = 1
    
    M = 3
    N,_= PopDec.shape

    y = np.zeros((N, M-1))
    for i in range(N):
        for j in range(M-1):
            dsum = 0
            y[i, j] = PopDec[i, j] * (2**(d+1) - 1)
            for m in range(d, -1, -1):
                dsum = dsum + 2**m
                if y[i, j] <= dsum:
                    y[i, j] = (y[i, j] - dsum + 2**m) / 2**m
                    break

    g = 100 * (kA + kB + np.sum((PopDec[:, M-1+kA:] - 0.5)**2 - np.cos(20 * np.pi * (PopDec[:, M-1+kA:] - 0.5)), axis=1))

    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(y * np.pi/2))), axis=1)) * \
             np.hstack((ones_mat, np.sin(y[:, M-2::-1] * np.pi/2)))

    return PopObj

def MMMOP5C(PopDec):
    PopDec = PopDec.reshape(1,6)
    kA = 1
    kB = 4
    c = 2
    d = 2
    
    M = 2
    N,_= PopDec.shape

    y = np.zeros((N, M-1))
    for i in range(N):
        for j in range(M-1):
            dsum = 0
            y[i, j] = PopDec[i, j] * (2**(d+1) - 1)
            for m in range(d, -1, -1):
                dsum = dsum + 2**m
                if y[i, j] <= dsum:
                    y[i, j] = (y[i, j] - dsum + 2**m) / 2**m
                    break

    g = 100 * (kA + kB + np.sum(np.cos(2 * np.pi * c * PopDec[:, M-1:M-1+kA]), axis=1) +
               np.sum((PopDec[:, M-1+kA:] - 0.5)**2 - np.cos(20 * np.pi * (PopDec[:, M-1+kA:] - 0.5)), axis=1))

    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(y * np.pi/2))), axis=1)) * \
             np.hstack((ones_mat, np.sin(y[:, M-2::-1] * np.pi/2)))

    return PopObj

def MMMOP5D(PopDec):
    PopDec = PopDec.reshape(1,7)
    kA = 1
    kB = 4
    c = 2
    d = 1
    
    M = 3
    N,_= PopDec.shape

    y = np.zeros((N, M-1))
    for i in range(N):
        for j in range(M-1):
            dsum = 0
            y[i, j] = PopDec[i, j] * (2**(d+1) - 1)
            for m in range(d, -1, -1):
                dsum = dsum + 2**m
                if y[i, j] <= dsum:
                    y[i, j] = (y[i, j] - dsum + 2**m) / 2**m
                    break

    g = 100 * (kA + kB + np.sum(np.cos(2 * np.pi * c * PopDec[:, M-1:M-1+kA]), axis=1) +
               np.sum((PopDec[:, M-1+kA:] - 0.5)**2 - np.cos(20 * np.pi * (PopDec[:, M-1+kA:] - 0.5)), axis=1))

    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(y * np.pi/2))), axis=1)) * \
             np.hstack((ones_mat, np.sin(y[:, M-2::-1] * np.pi/2)))

    return PopObj


def MMMOP6A(PopDec):
    PopDec = PopDec.reshape(1,2)
    kA = 0
    kB = 1
    c = 2

    M = 2
    N,_= PopDec.shape
    
    z = 2 * c * PopDec[:, M-1+kA:] - 2 * np.floor(c * PopDec[:, M-1+kA:]) - 1
    t = np.ones((N, kB))

    for i in range(kB):
        for j in range(M-1):
            t[:, i] = t[:, i] * np.sin(2 * np.pi * PopDec[:, j] + (i) * np.pi / kB)
    g = np.sum((z - t)**2, axis=1)
    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(PopDec[:, :M-1] * np.pi / 2))), axis=1)) * np.hstack((ones_mat, np.sin(PopDec[:, M-2::-1] * np.pi / 2)))

    return PopObj

def MMMOP6B(PopDec):
    PopDec = PopDec.reshape(1,4)
    kA = 0
    kB = 2
    c = 2

    M = 3
    N,_= PopDec.shape
    
    
    z = 2 * c * PopDec[:, M-1+kA:] - 2 * np.floor(c * PopDec[:, M-1+kA:]) - 1
    t = np.ones((N, kB))

    for i in range(kB):
        for j in range(M-1):
            t[:, i] = t[:, i] * np.sin(2 * np.pi * PopDec[:, j] + (i) * np.pi / kB)
   
    g = np.sum((z - t)**2, axis=1)
    
    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(PopDec[:, :M-1] * np.pi / 2))), axis=1)) * np.hstack((ones_mat, np.sin(PopDec[:, M-2::-1] * np.pi / 2)))

    return PopObj

def MMMOP6C(PopDec):
    PopDec = PopDec.reshape(1,4)
    kA = 2
    kB = 1
    c = 2

    M = 2
    N,_= PopDec.shape
    
    y = (PopDec[:, M-1:M+kA-1] - 0.5) * 12
    z = 2 * c * PopDec[:, M-1+kA:] - 2 * np.floor(c * PopDec[:, M-1+kA:]) - 1
    t = np.ones((N, kB))

    for i in range(kB):
        for j in range(M-1):
            t[:, i] = t[:, i] * np.sin(2 * np.pi * PopDec[:, j] + (i) * np.pi / kB)


    g = np.sum((y[:, 0:1]**2 + y[:, 1:2] - 11)**2 + (y[:, 0:1] + y[:, 1:2]**2 - 7)**2, axis=1) + np.sum((z - t)**2, axis=1)
    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(PopDec[:, :M-1] * np.pi / 2))), axis=1)) * np.hstack((ones_mat, np.sin(PopDec[:, M-2::-1] * np.pi / 2)))

    return PopObj

def MMMOP6D(PopDec):
    PopDec = PopDec.reshape(1,5)
    kA = 2
    kB = 1
    c = 2

    M = 3
    N,_= PopDec.shape
    
    y = (PopDec[:, M-1:M+kA-1] - 0.5) * 12
    z = 2 * c * PopDec[:, M-1+kA:] - 2 * np.floor(c * PopDec[:, M-1+kA:]) - 1
    t = np.ones((N, kB))

    for i in range(kB):
        for j in range(M-1):
            t[:, i] = t[:, i] * np.sin(2 * np.pi * PopDec[:, j] + (i) * np.pi / kB)


    g = np.sum((y[:, 0:1]**2 + y[:, 1:2] - 11)**2 + (y[:, 0:1] + y[:, 1:2]**2 - 7)**2, axis=1) + np.sum((z - t)**2, axis=1)
    ones_mat = np.ones((np.size(g, 0), 1))
    PopObj = (1 + g) * np.fliplr(np.cumprod(np.hstack((ones_mat, np.cos(PopDec[:, :M-1] * np.pi / 2))), axis=1)) * np.hstack((ones_mat, np.sin(PopDec[:, M-2::-1] * np.pi / 2)))

    return PopObj

def mapfunction(x):
    # 小学坐标
    es = np.array([[3, 37], [42, 96], [45, 60], [50, 25], [83, 72], [98, 38]])
    # 初中坐标
    js = np.array([[40, 20], [51, 60], [95, 51]])
    # 便利店
    cs = np.array([[10, 55], [15, 15], [15, 78], [15, 88], [20, 23],
                   [20, 70], [32, 42], [35, 60], [40, 76], [52, 78],
                   [52, 96], [55, 33], [75, 27]])
    # 地铁口
    rs = np.array([[17.5, 82.5], [55.5, 82.5], [94.5, 6.5]])

    y = np.zeros(4)
    #print(x)
    distances_e = np.linalg.norm(es - x, axis=1)
    #print("distances_e",distances_e)
    y[0] = np.min(distances_e)
    distances_j = np.linalg.norm(js - x, axis=1)
    #print("distances_e",distances_j)
    y[1] = np.min(distances_j)
    distances_c = np.linalg.norm(cs - x, axis=1)
    #print("distances_e",distances_c)
    y[2] = np.min(distances_c)
    distances_r = np.linalg.norm(rs - x, axis=1)
    #print("distances_e",distances_r)
    y[3] = np.min(distances_r)
    return y


def g1(x, t):
    g = (x - t) ** 2
    return g

def g2(x, t):
    g = 2 * (x - t)**2 + np.sin(2 * np.pi * (x - t))**2
    return g

def g3(x, t):
    g = 4 - (x - t) - 4/np.exp(100 * (x - t) ** 2)
    return g

def g4(x, t):
    g = np.sqrt((x - t) ** 2) + np.sin(2 * np.pi * (x - t)) ** 2
    return g

def g5(x, t):
    g = np.exp(np.log(2) * (x - t) ** 2) * (np.sin(6 * np.pi * (x - t)) ** 2) + (x - t) ** 2
    return g

def SMMOP1(X):
    X=X.reshape(1,100)
    M = 2
    theta = 0.1
    num_p = 4  # 'np' renamed to 'num_p' to avoid conflict with numpy alias
    N, D = X.shape
    S = int(np.ceil(theta * (D - M)))
    g = np.zeros((N, num_p))
    for i_Python in range(num_p):
        i_MATLAB = i_Python + 1
        # Indices adjusted for zero-based indexing in Python
        idx1_start = M - 1
        idx1_end = M - 1 + (i_MATLAB - 1) * S - 1
        if idx1_start <= idx1_end:
            idx1 = np.arange(idx1_start, idx1_end + 1, dtype=int)
        else:
            idx1 = np.array([], dtype=int)
        idx2_start = M - 1 + i_MATLAB * S
        idx2 = np.arange(idx2_start, D, dtype=int)
        idx_b = np.concatenate((idx1, idx2))
        b = X[:, idx_b]
        # Indices for g1
        idx_g1_start = M - 1 + (i_MATLAB - 1) * S
        idx_g1_end = M - 1 + i_MATLAB * S - 1
        if idx_g1_start <= idx_g1_end:
            idx_g1 = np.arange(idx_g1_start, idx_g1_end + 1, dtype=int)
            X_g1 = X[:, idx_g1]
            g1_vals = g1(X_g1, np.pi / 3)
        else:
            g1_vals = np.zeros((N, 0))
        # Indices for g2
        X_g2 = b
        g2_vals = g2(X_g2, 0)
        g[:, i_Python] = np.sum(g1_vals, axis=1) + np.sum(g2_vals, axis=1)
    min_g = np.min(g, axis=1)
    factor = 1 + min_g / (D - M + 1)
    factor = factor.reshape(N, 1)
    factor = np.tile(factor, (1, M))
    # Compute cumulative product and flip
    ones_col = np.ones((N, 1))
    if M - 1 > 0:
        X_subset = X[:, :M - 1]
        cumprod_array = np.cumprod(np.hstack([ones_col, X_subset]), axis=1)
    else:
        cumprod_array = ones_col
    cumprod_array = np.fliplr(cumprod_array)
    # Compute the reversed and complemented X values
    if M - 1 > 0:
        reverse_X = 1 - X[:, M - 2::-1]
        ones_reverse_X = np.hstack([ones_col, reverse_X])
    else:
        ones_reverse_X = ones_col
    PopObj = factor * cumprod_array * ones_reverse_X
    return PopObj

def SMMOP2(X):
    X=X.reshape(1,100)
    M = 2
    theta = 0.1
    num_p = 4  # 'np' renamed to 'num_p' to avoid conflict with numpy alias
    N, D = X.shape
    S = int(np.ceil(theta * (D - M)))
    g = np.zeros((N, num_p))
    for i_Python in range(num_p):
        i_MATLAB = i_Python + 1
        # Indices adjusted for zero-based indexing in Python
        idx1_start = M - 1
        idx1_end = M - 1 + (i_MATLAB - 1) * S - 1
        if idx1_start <= idx1_end:
            idx1 = np.arange(idx1_start, idx1_end + 1, dtype=int)
        else:
            idx1 = np.array([], dtype=int)
        idx2_start = M - 1 + i_MATLAB * S
        idx2 = np.arange(idx2_start, D, dtype=int)
        idx_b = np.concatenate((idx1, idx2))
        b = X[:, idx_b]
        # Indices for g1
        idx_g1_start = M - 1 + (i_MATLAB - 1) * S
        idx_g1_end = M - 1 + i_MATLAB * S - 1
        if idx_g1_start <= idx_g1_end:
            idx_g1 = np.arange(idx_g1_start, idx_g1_end + 1, dtype=int)
            X_g1 = X[:, idx_g1]
            g1_vals = g1(X_g1, np.pi / 3)
        else:
            g1_vals = np.zeros((N, 0))
        # Indices for g2
        X_g2 = b
        g2_vals = g3(X_g2, 0)
        g[:, i_Python] = np.sum(g1_vals, axis=1) + np.sum(g2_vals, axis=1)
    min_g = np.min(g, axis=1)
    factor = 1 + min_g / (D - M + 1)
    factor = factor.reshape(N, 1)
    factor = np.tile(factor, (1, M))
    # Compute cumulative product and flip
    ones_col = np.ones((N, 1))
    if M - 1 > 0:
        X_subset = X[:, :M - 1]
        cumprod_array = np.cumprod(np.hstack([ones_col, X_subset]), axis=1)
    else:
        cumprod_array = ones_col
    cumprod_array = np.fliplr(cumprod_array)
    # Compute the reversed and complemented X values
    if M - 1 > 0:
        reverse_X = 1 - X[:, M - 2::-1]
        ones_reverse_X = np.hstack([ones_col, reverse_X])
    else:
        ones_reverse_X = ones_col
    PopObj = factor * cumprod_array * ones_reverse_X
    return PopObj

def SMMOP3(X):
    X=X.reshape(1,100)
    M = 2
    theta = 0.1
    num_p = 4  # 'np' renamed to 'num_p' to avoid conflict with numpy alias
    N, D = X.shape
    S = int(np.ceil(theta * (D - M)))
    g = np.zeros((N, num_p))
    for i_Python in range(num_p):
        i_MATLAB = i_Python + 1
        # Indices adjusted for zero-based indexing in Python
        idx1_start = M - 1
        idx1_end = M - 1 + (i_MATLAB - 1) * S - 1
        if idx1_start <= idx1_end:
            idx1 = np.arange(idx1_start, idx1_end + 1, dtype=int)
        else:
            idx1 = np.array([], dtype=int)
        idx2_start = M - 1 + i_MATLAB * S
        idx2 = np.arange(idx2_start, D, dtype=int)
        idx_b = np.concatenate((idx1, idx2))
        b = X[:, idx_b]
        # Indices for g1
        idx_g1_start = M - 1 + (i_MATLAB - 1) * S
        idx_g1_end = M - 1 + i_MATLAB * S - 1
        if idx_g1_start <= idx_g1_end:
            idx_g1 = np.arange(idx_g1_start, idx_g1_end + 1, dtype=int)
            X_g1 = X[:, idx_g1]
            g1_vals = g2(X_g1, np.pi / 3)
        else:
            g1_vals = np.zeros((N, 0))
        # Indices for g2
        X_g2 = b
        g2_vals = g3(X_g2, 0)
        g[:, i_Python] = np.sum(g1_vals, axis=1) + np.sum(g2_vals, axis=1)
    min_g = np.min(g, axis=1)
    factor = 1 + min_g / (D - M + 1)
    factor = factor.reshape(N, 1)
    factor = np.tile(factor, (1, M))
    # Compute cumulative product and flip
    ones_col = np.ones((N, 1))
    if M - 1 > 0:
        X_subset = X[:, :M - 1]
        cumprod_array = np.cumprod(np.hstack([ones_col, X_subset]), axis=1)
    else:
        cumprod_array = ones_col
    cumprod_array = np.fliplr(cumprod_array)
    # Compute the reversed and complemented X values
    if M - 1 > 0:
        reverse_X = 1 - X[:, M - 2::-1]
        ones_reverse_X = np.hstack([ones_col, reverse_X])
    else:
        ones_reverse_X = ones_col
    PopObj = factor * cumprod_array * ones_reverse_X
    return PopObj

def SMMOP4(X):
    X=X.reshape(1,100)
    M = 2
    theta = 0.1
    num_p = 4  # 'np' renamed to 'num_p' to avoid conflict with numpy alias
    N, D = X.shape
    S = int(np.ceil(theta * (D - M)))
    g = np.zeros((N, num_p))
    for i_Python in range(num_p):
        i_MATLAB = i_Python + 1
        # Indices adjusted for zero-based indexing in Python
        idx1_start = M - 1
        idx1_end = M - 1 + (i_MATLAB - 1) * S - 1
        if idx1_start <= idx1_end:
            idx1 = np.arange(idx1_start, idx1_end + 1, dtype=int)
        else:
            idx1 = np.array([], dtype=int)
        idx2_start = M - 1 + i_MATLAB * S
        idx2 = np.arange(idx2_start, D, dtype=int)
        idx_b = np.concatenate((idx1, idx2))
        b = X[:, idx_b]
        # Indices for g1
        idx_g1_start = M - 1 + (i_MATLAB - 1) * S
        idx_g1_end = M - 1 + i_MATLAB * S - 1
        if idx_g1_start <= idx_g1_end:
            idx_g1 = np.arange(idx_g1_start, idx_g1_end + 1, dtype=int)
            X_g1 = X[:, idx_g1]
            g1_vals = g1(X_g1, np.pi / 3)
        else:
            g1_vals = np.zeros((N, 0))
        # Indices for g2
        X_g2 = b
        g2_vals = g4(X_g2, 0)
        g[:, i_Python] = np.sum(g1_vals, axis=1) + np.sum(g2_vals, axis=1)
    min_g = np.min(g, axis=1)
    factor = 1 + min_g / (D - M + 1)
    factor = factor.reshape(N, 1)
    factor = np.tile(factor, (1, M))
    ones_col = np.ones((N, 1))
    if M - 1 > 0:
        X_subset = X[:, :M - 1]
        cumprod_array = np.cumprod(np.hstack([ones_col, 1 - np.cos(X_subset * np.pi / 2)]), axis=1)
    else:
        cumprod_array = ones_col
    cumprod_array = np.fliplr(cumprod_array)
    if M - 1 > 0:
        reverse_X = 1 - np.sin(X[:, M - 2::-1] * np.pi / 2)
        ones_reverse_X = np.hstack([ones_col, reverse_X])
    else:
        ones_reverse_X = ones_col
    PopObj = factor * cumprod_array * ones_reverse_X
    return PopObj

def SMMOP5(X):
    X=X.reshape(1,100)
    M = 2
    theta = 0.1
    num_p = 4  # 'np' renamed to 'num_p' to avoid conflict with numpy alias
    N, D = X.shape
    S = int(np.ceil(theta * (D - M)))
    g = np.zeros((N, num_p))
    for i_Python in range(num_p):
        i_MATLAB = i_Python + 1
        # Indices adjusted for zero-based indexing in Python
        idx1_start = M - 1
        idx1_end = M - 1 + (i_MATLAB - 1) * S - 1
        if idx1_start <= idx1_end:
            idx1 = np.arange(idx1_start, idx1_end + 1, dtype=int)
        else:
            idx1 = np.array([], dtype=int)
        idx2_start = M - 1 + i_MATLAB * S
        idx2 = np.arange(idx2_start, D, dtype=int)
        idx_b = np.concatenate((idx1, idx2))
        b = X[:, idx_b]
        # Indices for g1
        idx_g1_start = M - 1 + (i_MATLAB - 1) * S
        idx_g1_end = M - 1 + i_MATLAB * S - 1
        if idx_g1_start <= idx_g1_end:
            idx_g1 = np.arange(idx_g1_start, idx_g1_end + 1, dtype=int)
            X_g1 = X[:, idx_g1]
            g1_vals = g1(X_g1, np.pi / 3)
        else:
            g1_vals = np.zeros((N, 0))
        # Indices for g2
        X_g2 = b
        g2_vals = g5(X_g2, 0)
        g[:, i_Python] = np.sum(g1_vals, axis=1) + np.sum(g2_vals, axis=1)
    min_g = np.min(g, axis=1)
    factor = 1 + min_g / (D - M + 1)
    factor = factor.reshape(N, 1)
    factor = np.tile(factor, (1, M))
    ones_col = np.ones((N, 1))
    if M - 1 > 0:
        X_subset = X[:, :M - 1]
        cumprod_array = np.cumprod(np.hstack([ones_col, 1 - np.cos(X_subset * np.pi / 2)]), axis=1)
    else:
        cumprod_array = ones_col
    cumprod_array = np.fliplr(cumprod_array)
    if M - 1 > 0:
        reverse_X = 1 - np.sin(X[:, M - 2::-1] * np.pi / 2)
        ones_reverse_X = np.hstack([ones_col, reverse_X])
    else:
        ones_reverse_X = ones_col
    PopObj = factor * cumprod_array * ones_reverse_X
    return PopObj

def SMMOP6(X):
    X=X.reshape(1,100)
    M = 2
    theta = 0.1
    num_p = 4  # 'np' renamed to 'num_p' to avoid conflict with numpy alias
    N, D = X.shape
    S = int(np.ceil(theta * (D - M)))
    g = np.zeros((N, num_p))
    for i_Python in range(num_p):
        i_MATLAB = i_Python + 1
        # Indices adjusted for zero-based indexing in Python
        idx1_start = M - 1
        idx1_end = M - 1 + (i_MATLAB - 1) * S - 1
        if idx1_start <= idx1_end:
            idx1 = np.arange(idx1_start, idx1_end + 1, dtype=int)
        else:
            idx1 = np.array([], dtype=int)
        idx2_start = M - 1 + i_MATLAB * S
        idx2 = np.arange(idx2_start, D, dtype=int)
        idx_b = np.concatenate((idx1, idx2))
        b = X[:, idx_b]
        # Indices for g1
        idx_g1_start = M - 1 + (i_MATLAB - 1) * S
        idx_g1_end = M - 1 + i_MATLAB * S - 1
        if idx_g1_start <= idx_g1_end:
            idx_g1 = np.arange(idx_g1_start, idx_g1_end + 1, dtype=int)
            X_g1 = X[:, idx_g1]
            g1_vals = g2(X_g1, np.pi / 3)
        else:
            g1_vals = np.zeros((N, 0))
        # Indices for g2
        X_g2 = b
        g2_vals = g4(X_g2, 0)
        g[:, i_Python] = np.sum(g1_vals, axis=1) + np.sum(g2_vals, axis=1)
    min_g = np.min(g, axis=1)
    factor = 1 + min_g / (D - M + 1)
    factor = factor.reshape(N, 1)
    factor = np.tile(factor, (1, M))
    ones_col = np.ones((N, 1))
    if M - 1 > 0:
        X_subset = X[:, :M - 1]
        cumprod_array = np.cumprod(np.hstack([ones_col, 1 - np.cos(X_subset * np.pi / 2)]), axis=1)
    else:
        cumprod_array = ones_col
    cumprod_array = np.fliplr(cumprod_array)
    if M - 1 > 0:
        reverse_X = 1 - np.sin(X[:, M - 2::-1] * np.pi / 2)
        ones_reverse_X = np.hstack([ones_col, reverse_X])
    else:
        ones_reverse_X = ones_col
    PopObj = factor * cumprod_array * ones_reverse_X
    return PopObj

def SMMOP7(X):
    X=X.reshape(1,100)
    M = 2
    theta = 0.1
    num_p = 4  # 'np' renamed to 'num_p' to avoid conflict with numpy alias
    N, D = X.shape
    S = int(np.ceil(theta * (D - M)))
    g = np.zeros((N, num_p))
    for i_Python in range(num_p):
        i_MATLAB = i_Python + 1
        # Indices adjusted for zero-based indexing in Python
        idx1_start = M - 1
        idx1_end = M - 1 + (i_MATLAB - 1) * S - 1
        if idx1_start <= idx1_end:
            idx1 = np.arange(idx1_start, idx1_end + 1, dtype=int)
        else:
            idx1 = np.array([], dtype=int)
        idx2_start = M - 1 + i_MATLAB * S
        idx2 = np.arange(idx2_start, D, dtype=int)
        idx_b = np.concatenate((idx1, idx2))
        b = X[:, idx_b]
        # Indices for g1
        idx_g1_start = M - 1 + (i_MATLAB - 1) * S
        idx_g1_end = M - 1 + i_MATLAB * S - 1
        if idx_g1_start <= idx_g1_end:
            idx_g1 = np.arange(idx_g1_start, idx_g1_end + 1, dtype=int)
            X_g1 = X[:, idx_g1]
            g1_vals = g2(X_g1, np.pi / 3)
        else:
            g1_vals = np.zeros((N, 0))
        # Indices for g2
        X_g2 = b
        g2_vals = g5(X_g2, 0)
        g[:, i_Python] = np.sum(g1_vals, axis=1) + np.sum(g2_vals, axis=1)
    min_g = np.min(g, axis=1)
    factor = 1 + min_g / (D - M + 1)
    factor = factor.reshape(N, 1)
    factor = np.tile(factor, (1, M))
    ones_col = np.ones((N, 1))
    if M - 1 > 0:
        X_subset = X[:, :M - 1]
        cumprod_array = np.cumprod(np.hstack([ones_col, np.cos(X_subset * np.pi / 2)]), axis=1)
    else:
        cumprod_array = ones_col
    cumprod_array = np.fliplr(cumprod_array)
    if M - 1 > 0:
        reverse_X = np.sin(X[:, M - 2::-1] * np.pi / 2)
        ones_reverse_X = np.hstack([ones_col, reverse_X])
    else:
        ones_reverse_X = ones_col
    PopObj = factor * cumprod_array * ones_reverse_X
    return PopObj

def SMMOP8(X):
    X=X.reshape(1,100)
    M = 2
    theta = 0.1
    num_p = 4  # 'np' renamed to 'num_p' to avoid conflict with numpy alias
    N, D = X.shape
    S = int(np.ceil(theta * (D - M)))
    g = np.zeros((N, num_p))
    for i_Python in range(num_p):
        i_MATLAB = i_Python + 1
        # Indices adjusted for zero-based indexing in Python
        idx1_start = M - 1
        idx1_end = M - 1 + (i_MATLAB - 1) * S - 1
        if idx1_start <= idx1_end:
            idx1 = np.arange(idx1_start, idx1_end + 1, dtype=int)
        else:
            idx1 = np.array([], dtype=int)
        idx2_start = M - 1 + i_MATLAB * S
        idx2 = np.arange(idx2_start, D, dtype=int)
        idx_b = np.concatenate((idx1, idx2))
        b = X[:, idx_b]
        # Indices for g1
        idx_g1_start = M - 1 + (i_MATLAB - 1) * S
        idx_g1_end = M - 1 + i_MATLAB * S - 1
        if idx_g1_start <= idx_g1_end:
            idx_g1 = np.arange(idx_g1_start, idx_g1_end + 1, dtype=int)
            X_g1 = X[:, idx_g1]
            g1_vals = g4(X_g1, np.pi / 3)
        else:
            g1_vals = np.zeros((N, 0))
        # Indices for g2
        X_g2 = b
        g2_vals = g5(X_g2, 0)
        g[:, i_Python] = np.sum(g1_vals, axis=1) + np.sum(g2_vals, axis=1)
    min_g = np.min(g, axis=1)
    factor = 1 + min_g / (D - M + 1)
    factor = factor.reshape(N, 1)
    factor = np.tile(factor, (1, M))
    ones_col = np.ones((N, 1))
    if M - 1 > 0:
        X_subset = X[:, :M - 1]
        cumprod_array = np.cumprod(np.hstack([ones_col, np.cos(X_subset * np.pi / 2)]), axis=1)
    else:
        cumprod_array = ones_col
    cumprod_array = np.fliplr(cumprod_array)
    if M - 1 > 0:
        reverse_X = np.sin(X[:, M - 2::-1] * np.pi / 2)
        ones_reverse_X = np.hstack([ones_col, reverse_X])
    else:
        ones_reverse_X = ones_col
    PopObj = factor * cumprod_array * ones_reverse_X
    return PopObj
