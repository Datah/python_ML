import math


def extend_vals(vals):
    return (1,) + vals

def extend_vals_vert(vals):
    return ((1,),) + vals


def mean(vals):
    mean = sum(v for v in vals) / len(vals)
    return mean


def standard_deviation(vals):
    meanv = mean(vals)
    stdev = math.sqrt(sum((v - meanv)**2 for v in vals)/(len(vals) - 1))
    return stdev


def stats(pts):
    l = len(pts[0][0])
    nstdevs = []
    navgs = []

    for n in range(l):
        xnvs = [x[n] for x, y in pts]
        meanv = mean(xnvs)
        stdev = standard_deviation(xnvs)
        navgs.append(meanv)
        nstdevs.append(stdev)
    return {"stdevs": nstdevs, "avgs": navgs}


def standardize(pts):
    lstats = stats(pts)

    return {"points": standardize_given(pts, lstats), "stats": lstats}


def standardize_given(pts, std_stats):
    navgs = std_stats["avgs"]
    nstdevs = std_stats["stdevs"]
    newpts = []
    for x, y in pts:
        newx = []
        for n in range(len(x)):
            newx.append((x[n] - navgs[n]) / nstdevs[n])
        newpts.append((tuple(newx), y))
    return tuple(newpts)


def standardize_tuple_given(x, std_stats):
    navgs = std_stats["avgs"]
    nstdevs = std_stats["stdevs"]
    newx = []
    for n in range(len(x)):
        newx.append((x[n] - navgs[n]) / nstdevs[n])
    return tuple(newx)


def dot(a, b):
    if not (type(a) == tuple or type(a) == list) or not (type(b) == tuple or type(b) == list):
        print("Error doing dot product for {} with {}".format(a, b))
        raise TypeError
    return sum(a[n] * b[n] for n in range(0, len(a)))


def mat_mult(m, n):
    mat = []
    for j in range(len(m)):
        row = m[j]
        ncols = len(n[0])
        cols = [[r[c] for r in n] for c in range(ncols)]
        prn = [dot(row, cols[c]) for c in range(ncols)]
        mat.append(tuple(prn))
    return tuple(mat)


def matrix(tuple):
    return (tuple,)


def transpose(matrix):
    cols = len(matrix[0])
    new_matrix = tuple(tuple(matrix[m][n] for m in range(len(matrix))) for n in range(cols))
    return new_matrix


def init_matrix(n, m, func):
    return tuple(tuple(func(i, j) for i in range(m)) for j in range(n))


def val_matrix(n, m, val = 0):
    return init_matrix(n, m, const_func(val))


def const_func(val):
    def cfunc(i, j):
        return val
    return cfunc


print("Transposition: {}".format(transpose(((1, 2, 3), (4, 5, 6)))))


print("Stdev: {}".format(standard_deviation((1, 2, 3, 1, 2, 3))))