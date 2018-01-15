import math

from regression_core import extend_vals, dot, logistic_fcost, least_squares_fcost


def logistic_derivs(logistic_func, coeffs):
    def generated_deriv(n):
        def n_deriv(vals):
            evals = extend_vals(vals)
            if type(vals) != type(coeffs) or len(evals) != len(coeffs):
                print("Error in list sizes. Vals {} // Coeffs {}".format(evals, coeffs))
                raise IndexError
            prod = dot(evals, coeffs)
            deriv = -logistic_func(vals)**2 * (-evals[n] * math.exp(-prod))
            return deriv
        return n_deriv
    return tuple(generated_deriv(n) for n in range(len(coeffs)))


def logistic_function(coeffs):
    def generated(vals):
        vals = extend_vals(vals)
        if len(vals) != len(coeffs):
            raise IndexError
        prod = dot(vals, coeffs)
        return 1/(1 + math.exp(-prod))
    funcmap = {"func": generated, "params": coeffs, "derivs": logistic_derivs(generated, coeffs)}
    costmap = logistic_fcost(funcmap)
    funcmap.update(costmap)
    return funcmap


def linear_derivs(coeffs):
    def generated_deriv(n):
        def n_deriv(vals):
            return extend_vals(vals)[n]
        return n_deriv
    return tuple(generated_deriv(n) for n in range(len(coeffs)))


def linear_function(coeffs):
    def generated(vals):
        vals = extend_vals(vals)
        if len(vals) != len(coeffs):
            raise IndexError
        return sum(c*v for (c, v) in zip(coeffs, vals))
    funcmap = {"func": generated, "params": coeffs, "derivs": linear_derivs(coeffs)}
    costmap = least_squares_fcost(funcmap)
    funcmap.update(costmap)
    return funcmap