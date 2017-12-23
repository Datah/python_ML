import math
import nn_function_defs

def square_diff(nn, x, y):
    lfunc = funcmap["func"]
    deriv = funcmap["derivs"]
    def func(x, y):
        return (lfunc(x) - y)**2
    def deriv(x, r, c, l):
        fderivs = funcmap["derivs"]
        return -2*(lfunc(x) - y) * fderivs(x, r, c, l)
    return nn_function_defs.fmap(func, deriv)