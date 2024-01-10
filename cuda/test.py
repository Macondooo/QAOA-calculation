import numpy as np
import cupy as cp

p=3

def idx2arr(idx,p):
    a = cp.zeros(2 * p + 1)
    for i in range(2 * p + 1):
        tmp = idx % 2
        if tmp == 0:
            a[i] = 1
        else: a[i] = -1
        idx = idx // 2

    return a

indices = cp.arange(1 << (2 * p + 1))
a = cp.array([idx2arr(idx,p) for idx in indices]).astype(dtype=np.int8)
pre_E_H = a[:, None, None, None] * a[:, None, None] + a[:, None, None, None] * a[:, None] +  a[:, None, None, None] * a + a[:, None, None] * a[:, None]#(ab+ac+ad+bc)

tmp=cp.ones(2*p+1)
E_H = cp.einsum("m,ijklm->ijkl",tmp, pre_E_H)

