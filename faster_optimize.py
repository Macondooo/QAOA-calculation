from scipy.optimize import minimize, Bounds
from math import pi, cos, sin
import numpy as np
import cmath

global finish, p, pre_E_G, pre_E_H

def callback(intermediate_result):
    global finish
    finish+=1
    print(finish)
    print(intermediate_result)
    print(0.5-objective(intermediate_result))
    
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
        a[i] = idx % 2
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
    for idx in range(1 << (2 * p + 1)):
        a = idx2arr(idx)
        f_pre[idx] = func(a, beta)
    # pre-calculation of E_G, E_H
    E_G = np.einsum("k,ijk->ij", Gamma, pre_E_G)
    E_G = np.exp(-1j * E_G)
    E_H = np.einsum("m,ijklm->ijkl", Gamma, pre_E_H)
    E_H = np.exp(-1j * E_H)

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
    for a_idx in range(1 << (2 * p + 1)):
        for b_idx in range(1 << (2 * p + 1)):
            for c_idx in range(1 << (2 * p + 1)):
                a = idx2arr(a_idx)
                b = idx2arr(b_idx)
                c = idx2arr(c_idx)

                res = res + a[p] * b[p] * f_pre[a_idx] * f_pre[b_idx] * f_pre[
                    c_idx
                ] * G[p, a_idx] * G[p, b_idx] * G[p - 1, c_idx] * cmath.exp(
                    complex(0, -np.dot(Gamma, a * b + a * c + b * c))
                )

    # second kind of edge
    for a_idx in range(1 << (2 * p + 1)):
        for b_idx in range(1 << (2 * p + 1)):
            a = idx2arr(a_idx)
            b = idx2arr(b_idx)

            res = res + a[p] * b[p] * f_pre[a_idx] * f_pre[b_idx] * H[p, a_idx] * H[
                p, b_idx
            ] * cmath.exp(complex(0, -np.dot(Gamma, a * b)))

    #
    return 0.25 * res.real
    # return res


# main
finish=0
p=2
# some pre-calculation
pre_E_G = np.empty((1 << (2 * p + 1), 1 << (2 * p + 1), 2 * p + 1), dtype=complex)
for idx in range(1 << (2 * p + 1)):
    a = idx2arr(idx)
    for sub_idx in range(1 << (2 * p + 1)):
        b = idx2arr(sub_idx)
        pre_E_G[idx, sub_idx] = a * b

pre_E_H = np.empty(
    (
        1 << (2 * p + 1),
        1 << (2 * p + 1),
        1 << (2 * p + 1),
        1 << (2 * p + 1),
        2 * p + 1,
    ),
    dtype=complex,
)
for idx in range(1 << (2 * p + 1)):
    for b_idx in range(1 << (2 * p + 1)):
        for c_idx in range(1 << (2 * p + 1)):
            for d_idx in range(1 << (2 * p + 1)):
                a = idx2arr(idx)
                b = idx2arr(b_idx)
                c = idx2arr(c_idx)
                d = idx2arr(d_idx)
                pre_E_H[idx, b_idx, c_idx, d_idx] = a * b + a * c + a * d + b * c
print("pre-calculations done")

gamma0 = np.full(p, pi / 6)  # (gamma_1,gamma_2,...,gamma_p)
beta0 = np.full(p, pi / 3)  # (beta_1,beta_2,...,beta_p)
# initial points
x0 = np.hstack((gamma0, beta0))
# bounds on gamma and beta
bounds = [[0, 2 * pi]] * p + [[0, pi]] * p
# optimize the objective function
res = minimize(objective, x0, bounds=bounds,callback=callback)


# output results
with open("output.txt", "w") as f:
    f.write("layers p = %d\n" % (p))

    f.write("optimized gamma: ")
    f.write(str(res.x[:p]) + "\n")

    f.write("optimized beta: ")
    f.write(str(res.x[p:]) + "\n")

    f.write("optimized function: ")
    f.write(str(0.5 - res.fun) + "\n")

    f.write("Success or not: ")
    f.write(str(res.success) + "\n")

    f.write("Reasons for stopping: ")
    f.write(res.message + "\n\n")
