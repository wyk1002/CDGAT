import numpy as np

def CR_calculation(obtained_ps, reference_ps):
    # Calculate the cover rate of the obtained Pareto set

    # obtained_ps: obtained Pareto set (population_size x n_var)
    # reference_ps: reference Pareto set (num_of_solutions_in_reference_ps x n_var)

    n_var = reference_ps.shape[1]
    # find the maximum and minimum in each dimension of obtained Pareto set
    obtained_min = np.min(obtained_ps, axis=0)
    obtained_max = np.max(obtained_ps, axis=0)
    # find the maximum and minimum in each dimension of reference Pareto set
    reference_min = np.min(reference_ps, axis=0)
    reference_max = np.max(reference_ps, axis=0)

    kesi = np.zeros(n_var)
    for i in range(n_var):
        if reference_max[i] == reference_min[i]:
            kesi[i] = 1
        elif obtained_min[i] >= reference_max[i] or reference_min[i] >= obtained_max[i]:
            kesi[i] = 0
        else:
            kesi[i] = ((min(obtained_max[i],reference_max[i])-max(obtained_min[i],reference_min[i])) / (reference_max[i] - reference_min[i])) ** 2

    CR = np.power(np.prod(kesi), 1 / (2 * n_var))

    return CR

def Hypervolume_calculation(pf, repoint):
    # Calculate the hypervolume of the obtained Pareto front

    # pf: obtained Pareto front (population_size x n_obj)
    # repoint: reference point (1 x n_obj), depending on the test function
    #print(pf.shape)
    popsize = pf.shape[0]
    #print(np.argsort(pf[:, 0]))
    sorted_pf = pf[np.argsort(pf[:, 0]), :]
    #print(repoint.shape)
    #print(sorted_pf.shape)
    pointset = np.concatenate((repoint, sorted_pf), axis=0)#np.vstack((repoint, sorted_pf))
    hyp = 0
    #print("pointset",pointset)
    for i in range(popsize):
        cubei = (pointset[0][0] - pointset[i+1][0]) * (pointset[i][1] - pointset[i+1][1])
        hyp += cubei

    return hyp

def IGD_calculation(obtained_ps, reference_ps):
    # Calculate the IGD of the obtained Pareto set

    # obtained_ps: obtained Pareto set (population_size x n_var)
    # reference_ps: reference Pareto set (num_of_solutions_in_reference_ps x n_var)

    n_ref = reference_ps.shape[0]
    obtained_to_ref = np.zeros(n_ref)

    for i in range(n_ref):
        ref_m = np.tile(reference_ps[i, :], (obtained_ps.shape[0], 1))
        d = obtained_ps - ref_m  # Calculate the differences between obtained_ps and reference_ps
        D = np.sum(np.abs(d)**2, axis=1)**0.5  # Calculate the distance between obtained_ps and reference_ps
        obtained_to_ref[i] = np.min(D)

    IGD = np.sum(obtained_to_ref) / n_ref

    return IGD
