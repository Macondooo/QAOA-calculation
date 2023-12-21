import numpy as np
from scipy.optimize import minimize

# a = np.full(3, 1)
# print(a)
# b = -a
# print(b)
# print(a * b)
# print(np.dot(a, b))


def obj(x):
    return np.linalg.norm(x[0] + x[1] * 1j - (1 + 1j)*(1 + 1j))


res = minimize(obj, (0, 0))

print(res.x)

# a=np.zeros(3)
# a[1]=1+1j