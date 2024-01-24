from scipy.optimize import minimize, Bounds
from math import pi, cos, sin
import numpy as np
import cmath
import os


def callback(intermediate_result):
    global finish
    finish+=1
    with open(out_file, "a") as f:
        f.write("Iterations %d: " %(finish,))
        f.write("gamma, beta: %s, %s, " %(str(intermediate_result.x[0:p]), str(intermediate_result.x[p:])))
        f.write("cut fraction: %s\n" %(str(0.5-intermediate_result.fun),))
        f.flush()
    
    # percentage = round( progress_bar.finish_tasks_number /  1000 * 100)
    # print("\rprogress: {}%: ".format(percentage), " " * (percentage // 2), end="")
    # sys.stdout.flush()

def func(a, beta):
    # a: np.array of size (2p+1), (a_1,a_2,...,a_p,a_0,a_{-p},...,a_{-1})
    # beta: np.array of size (p), (beta_1,beta_2,...,beta_p)
    res = 0.5
    for i in range(p):
        if a[i] == a[i + 1]:
            res = res * cos(beta[i])
        else:
            res = res * complex(0, sin(beta[i]))

    for i in range(p, 2 * p):
        if a[i] == a[i + 1]:
            res = res * cos(-beta[2 * p - i - 1])
        else:
            res = res * complex(0, sin(-beta[2 * p - i - 1]))

    return res

# convert an integer to (2p+1) bin array
def idx2arr(idx):
    a = np.zeros(2 * p + 1)
    for i in range(2 * p + 1):
        tmp = idx % 2
        if tmp == 0:
            a[i] = 1
        else: a[i] = -1
        idx = idx // 2

    return a

# define the objective function
def objective(x):
    print("called")
    gamma = x[0:p]
    beta = x[p:]
    Gamma = np.hstack((gamma, [0], -gamma[::-1]))

    # the iteration symbol, with initializing G[0,a] and H[0,a] to 1
    # the configuration basis a, is numerically represented in lexicographic order
    G = np.full((p + 1, 1 << (2 * p + 1)), 1, dtype=complex)
    H = np.full((p + 1, 1 << (2 * p + 1)), 1, dtype=complex)

    # pre-calculation of f(a)
    f_pre = np.zeros(1 << (2 * p + 1), dtype=complex)
    a = np.zeros((1 << (2 * p + 1), 2 * p + 1))
    for idx in range(1 << (2 * p + 1)):
        a[idx] = idx2arr(idx)
        f_pre[idx] = func(a[idx], beta)

    # pre-calculation of E_G, E_H
    E_G = np.einsum("k,ijk->ij", Gamma, pre_E_G)
    E_G = np.exp(-1j * E_G)
    E_H = np.einsum("m,ijklm->ijkl", Gamma, pre_E_H)
    E_H = np.exp(-1j * E_H)
    E_1 = np.einsum("m,ijkm->ijk", Gamma, pre_E_1)
    E_1 = np.exp(-1j * E_1)
    E_2 = E_G

    # calculation of G and H
    for i in range(p):
        # calculation of G
        G[i + 1] = np.einsum("i,i,ji->j", f_pre, H[i], E_G) ** 2
        # calculation of H
        H[i + 1] = np.einsum(
            "j,k,l,j,k,l,ijkl->i", f_pre, f_pre, f_pre, G[i], G[i], H[i], E_H
        )

    res = 0
    # first kind of edge
    res+=np.einsum("i,j,i,j,k,i,j,k,ijk->", a[:,p], a[:,p], f_pre, f_pre, f_pre, G[p], G[p], G[p-1], E_1)
    # second kind of edge
    res+=np.einsum("i,j,i,j,i,j,ij->", a[:,p], a[:,p], f_pre, f_pre, H[p], H[p], E_2)

    return 0.25 * res.real


#----------------------------------global parameters and path------------------------------------------
p = 2
out_file = "output_p2.txt"
if os.path.exists(out_file):
	os.remove(out_file)
# ---------------------------------initial points-------------------------------------------------------
# p = 1
# gamma=np.array([[0.6],[pi/2],[pi/2],[pi/4],[pi/6],[pi/8],[pi/4],[pi/6],[pi/8],[0],[pi/3]])
# beta=np.array([[pi/8],[pi/2],[pi/4],[pi/2],[pi/8],[pi/8],[pi/4],[pi/6],[0],   [0],[pi/3]])

# p = 2
gamma = np.array([[0.3817, 0.6655],[pi/2,pi/2],[pi/4,pi/4],[pi/8,pi/8],[pi/8,pi/6],[pi/8,pi/6]])  
beta = np.array([[0.4960, 0.2690], [pi/2,pi/2],[pi/4,pi/4],[pi/8,pi/8],[0,pi/3],[0,pi/4]])  

# p = 3
# gamma = np.array([[0.3297, 0.5688, 0.6406],[pi/8,pi/8,pi/8],[pi/6,pi/6,pi/6],[pi/4,pi/3,pi/3],[pi/8,pi/8,pi/8]])  
# beta =  np.array([[0.5500, 0.3675, 0.2109],[pi/8,pi/8,pi/8],[pi/6,pi/6,pi/6],[pi/8,pi/8,pi/8],[pi/3,pi/4,pi/8]])  

#---------------------------------preprocessing---------------------------------------------------------
# Generate arrays a, b, c, d for all indices
indices = np.arange(1 << (2 * p + 1))
a = np.array([idx2arr(idx) for idx in indices]).astype(dtype=np.int8)
# Perform element-wise multiplication and addition using vectorized operations
pre_E_G = a[:, None] * a #ab
pre_E_H = a[:, None, None, None] * a[:, None, None] + a[:, None, None, None] * a[:, None] +  a[:, None, None, None] * a + a[:, None, None] * a[:, None]#(ab+ac+ad+bc)
pre_E_1 = a[:, None, None] * a[:, None] + a[:, None, None] * a + a[:, None] * a #ab+ac+bc
print("pre-calculations done!")

#---------------------------------optimizing------------------------------------------------------------
with open(out_file, "a") as f:
    f.write("layers p = %d\n\n" % (p))
    f.flush()
    for gamma0, beta0 in zip(gamma, beta):
        # initial points
        x0 = np.hstack((gamma0, beta0))
        # bounds on gamma and beta
        bounds = [[0, 2 * pi]] * p + [[0, pi]] * p

        f.write("Initially:    ")
        f.write("gamma, beta: %s, %s, " %(str(gamma0),str(beta0)))
        f.write("cut fraction: %s\n" %(str(0.5-objective(x0)),))
        f.flush()

        # optimize the objective function
        finish = 0
        res = minimize(objective, x0, bounds=bounds,callback=callback)
        
        # output results
        # f.write("optimized gamma: %s\n" %(str(res.x[:p]),))
        # f.write("optimized beta: %s\n" %(str(res.x[p:]),))
        # f.write("optimized function: %s\n" %(str(0.5 - res.fun),))
        f.write("Success or not: %s\n" %(str(res.success),))
        f.write("Reasons for stopping: %s\n\n" %(res.message,))
        f.flush()
