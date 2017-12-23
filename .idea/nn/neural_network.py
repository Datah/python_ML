import math
import data_utils
import nn_function_defs
import random
import tensor_utils

def apply(ov, func):
    lfunc = func["func"]
    val = tuple((lfunc(x[0]),) for x in ov)
    return val


def nn_func(tensor, activations, std_vals):
    def func(x):
        lstd_vals = std_vals
        if not lstd_vals:
            lstd_vals = {"avgs": tuple(0 for n in len(x)), "stdevs": tuple(1 for n in len(x))}
        xlen = len(x)
        ilen = len(tensor[0][0])
        slen = len(lstd_vals["avgs"])
        layers = len(tensor)
        if xlen == slen == ilen - 1:
            x = data_utils.standardize_tuple_given(x, lstd_vals)
            x = data_utils.extend_vals(x)
            #print("Input vector for transpose {}".format(x))
            iv = data_utils.transpose(data_utils.matrix(x))
            #print("Transposed vector {}".format(iv))
            linear_vals = []
            for n in range(layers):
                matrix = tensor[n]
                mv = data_utils.mat_mult(matrix, iv)
                print("\nMultiplying matrix {} with vector {} \n\tResult {}".format(matrix, iv, mv))
                linear_vals.append(mv)
                afunc = activations[n]
                ov = apply(mv, afunc)
                iv = data_utils.extend_vals_vert(ov)
            return {"output": data_utils.transpose(ov)[0], "zvecs": linear_vals}
        else:
            raise IndexError
    return func


def nn_deriv(base_nn_func, tensor, activations, std_vals, gap):
    def grad(x, r, c, l):
        new_tensor = tensor_utils.incr_in_tensor(tensor, r, c, l, gap)
        #print("New tensor: {}".format(new_tensor))
        mod_nn_func = nn_func(new_tensor, activations, std_vals)
        new_out = mod_nn_func(x)["output"]
        base_out = base_nn_func(x)["output"]
        func_derivs_approx = tuple((new_out[n] - base_out[n]) / gap for n in range(len(new_out)))
        return func_derivs_approx
    return grad


def nn(tensor, activations, std_vals = False, deriv_gap = 0.05):
    func = nn_func(tensor, activations, std_vals)
    grad = nn_deriv(func, tensor, activations, std_vals, deriv_gap)
    return {"func": func, "coeffs": tensor, "fns": activations, "derivs": grad, "std_vals": std_vals}


pts = (((0.1, 0.8, 0.2), 0), ((0.4, 0.92, 0.35), 0), ((0.3, 0.89, 0.22), 0), ((0.2, 0.81, 0.27), 0),
       ((0.75, 0.4, 0.4), 1), ((0.7, 0.79, 0.4), 1), ((0.9, 0.52, 0.73), 1), ((0.6, 0.01, 0.99), 1))


std_dict = data_utils.standardize(pts)

x = (0.5, 0.2, 0.1)

#std = std_dict["stats"]

std = {"avgs": (0, 0, 0), "stdevs": (1, 1, 1)}


layers = []
layers.append(data_utils.init_matrix(5, 4, lambda n, m: random.randrange(0, 7)))
layers.append(data_utils.init_matrix(2, 6, lambda n, m: random.uniform(-1, 1)))

activations = (nn_function_defs.relu(), nn_function_defs.sigmoid())

network = nn(layers, activations, std)

deriv_ex = network["derivs"](x, 0, 0, 0)

print("Matrix: {}, Vector: {}".format(layers, x))

val = network["func"](x)

print("\n\n\nNNVal: {} // Deriv (row 0, col 0, layer 0): {}".format(val, deriv_ex))