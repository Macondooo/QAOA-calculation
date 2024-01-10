import cupy as cp
from math import pi, cos, sin
import sys

# define the objective function
def objective(x,p,pre_E_G,pre_E_H,pre_E_1):
    # print("called")
    gamma = x[0:p]
    beta = x[p:]
    Gamma = cp.hstack((gamma, [0], -gamma[::-1]))

    # the iteration symbol, with initializing G[0,a] and H[0,a] to 1
    # the configuration basis a, is numerically represented in lexicographic order
    G = cp.full((p + 1, 1 << (2 * p + 1)), 1, dtype=complex)
    H = cp.full((p + 1, 1 << (2 * p + 1)), 1, dtype=complex)

    # pre-calculation of f(a)
    f_pre = cp.zeros(1 << (2 * p + 1), dtype=complex)
    a = cp.zeros((1 << (2 * p + 1), 2 * p + 1))
    for idx in range(1 << (2 * p + 1)):
        a[idx] = idx2arr(idx,p)
        f_pre[idx] = func(a[idx], beta,p)

    # pre-calculation of E_G, E_H
    E_G = cp.einsum("k,ijk->ij", Gamma, pre_E_G)
    E_G = cp.exp(-1j * E_G)
    E_H = cp.einsum("m,ijklm->ijkl", Gamma, pre_E_H)
    E_H = cp.exp(-1j * E_H)
    E_1 = cp.einsum("m,ijkm->ijk", Gamma, pre_E_1)
    E_1 = cp.exp(-1j * E_1)
    E_2 = E_G

    # calculation of G and H
    for i in range(p):
        # calculation of G
        G[i + 1] = cp.einsum("i,i,ji->j", f_pre, H[i], E_G) ** 2
        # calculation of H
        H[i + 1] = cp.einsum(
            "j,k,l,j,k,l,ijkl->i", f_pre, f_pre, f_pre, G[i], G[i], H[i], E_H
        )

    res = 0
    # first kind of edge
    res+=cp.einsum("i,j,i,j,k,i,j,k,ijk->", a[:,p], a[:,p], f_pre, f_pre, f_pre, G[p], G[p], G[p-1], E_1)
    # second kind of edge
    res+=cp.einsum("i,j,i,j,i,j,ij->", a[:,p], a[:,p], f_pre, f_pre, H[p], H[p], E_2)

    # print(type(res.real))
    return float(0.25 * res.real)

# convert an integer to (2p+1) bin array
def idx2arr(idx,p):
    a = cp.zeros(2 * p + 1)
    for i in range(2 * p + 1):
        tmp = idx % 2
        if tmp == 0:
            a[i] = 1
        else: a[i] = -1
        idx = idx // 2

    return a

def func(a, beta,p):
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