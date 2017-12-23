import math

def fmap(func, deriv):
    return {"func": func, "deriv": deriv}


def relu():
    def func(x):
        return max(0, x)
    def grad(x):
        val = 0 if x < 0 else 1
        return val
    return fmap(func, grad)


def approx_relu():
    def func(x):
        return math.log(1 + math.exp(x))
    def grad(x):
        return math.exp(x) / (1 + math.exp(x))
    return fmap(func, grad)


def const():
    def func(x):
        return x
    def grad(x):
        return 1
    return fmap(func, grad)


def sigmoid():
    def func(x):
        return 1 / (1 + math.exp(-x))
    def grad(x):
        return math.exp(-x) * func(x)**2
    return fmap(func, grad)


def tanh():
    def func(x):
        return math.tanh(x)
    def grad(x):
        return 1 - func(x)**2
    return fmap(func, grad)
