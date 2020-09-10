import numpy as np

hs = np.random.randn(5,4)
print(hs)

a = np.array([0.8,0.1,0.03,0.05,0.02]).reshape(5,1)
print(a.shape)

ar = a.repeat(4, axis=1)
print(ar)

t = hs * ar
print(t)

c = np.sum(t, axis=0)
print(c)

# 미니배치 10개
hs = np.random.randn(10,5,4)
a = np.random.randn(10,5)
ar = a.reshape(10,5,1).repeat(4,axis=2)

c = np.sum(hs * ar, axis=1)
print(c)
print(c.shape)