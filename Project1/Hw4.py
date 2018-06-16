# Question 1
import numpy as np
from tqdm import tqdm
f = lambda x , y : 80* (x**4) + 0.01 * (y**6)
def grad_f(x, y):
    dx = 320 * x**3
    dy = 0.06 * y**5
    return dx , dy
# we know that the function is Convex define there for to find the
# minimun we will keep iterating until we over pass the minimun
def gradiant_decent(x,y, lr, max_iter=1_000):
    fx = f(x,y)
    for _ in tqdm(range(max_iter)):
        dx , dy = grad_f(x,y)
        x , y = [x-(lr *dx), y-(lr *dy)]
        fx_t = f(x,y)
        if fx_t>fx:
            break
    return fx_t, x , y
#%%