import numpy
from numpy import linalg as LA
import math


def line_search(f, x0, step_size, obj_tol, param_tol, max_iter, dir_selection_method='gd', init_step_len=1.0,
                slope_ratio=math.exp(0-4), back_track_factor=0.2):
    if dir_selection_method == 'gd':
        f_prev, grad_prev = f(x0)
    else:
        f_prev, grad_prev, hessian_prev = f(x0, True)

    x0_hist, x1_hist, f_hist = [x0[0]], [x0[1]], [f_prev]
    x_prev = x0

    print("f_prev: " + str(f_prev))
    print("df_prev: " + str(grad_prev))
    i = 0
    success = False
    while i <= max_iter:
        print("iteration number: " + str(i))

        if dir_selection_method == 'nt' or (i == 0 and dir_selection_method == 'bfgs'):
            direction = newton_dir(grad_prev, hessian_prev)
        elif dir_selection_method == 'bfgs':
            if i != 0:
                hessian_next = bfgs_dir(hessian_prev, param_change_bfgs, grad_change_bfgs)
                direction = newton_dir(grad_prev, hessian_next)
                hessian_prev = hessian_next
        else:
            direction = -grad_prev

        if dir_selection_method != 'gd':
            # backtracking
            step_size = init_step_len
            a, _ = f(x_prev + (step_size * direction))
            b, _ = f(x_prev) + (slope_ratio * step_size * grad_prev.transpose() * direction)
            sufficient_dec = (a <= b).all()
            while not sufficient_dec:
                step_size = back_track_factor * step_size
                a, _ = f(x_prev + (step_size * direction))
                b, _ = f(x_prev) + (slope_ratio * step_size * grad_prev.transpose() * direction)
                sufficient_dec = (a <= b).all()
        x_next = x_prev + (step_size * direction)

        if dir_selection_method == 'nt':
            f_next, grad_next, hessian_next = f(x_next, True)
        else:
            f_next, grad_next = f(x_next, False)

        print("current location: " + str(x_next))
        print("current objective value: " + str(f_next))

        param_change = LA.norm(x_next - x_prev)

        param_change_bfgs = numpy.matrix(x_next - x_prev)
        grad_change_bfgs = numpy.matrix(grad_next - grad_prev)

        obj_change = abs(f_next - f_prev)

        print("the current step length taken: " + str(param_change))
        print("current change in objective function value: " + str(obj_change))

        if (obj_change < obj_tol) or (param_change < param_tol):
            success = True
            break

        x_prev = x_next
        f_prev = f_next
        grad_prev = grad_next
        if dir_selection_method == 'nt':
            hessian_prev = hessian_next
        x0_hist.append(x_next[0])
        x1_hist.append(x_next[1])
        f_hist.append(f_next)
        i += 1

    return x_next, success, x0_hist, x1_hist, f_hist


def newton_dir(gradient, hessian):
    return LA.solve(hessian, -gradient)


def bfgs_dir(bk, param_change, grad_change):
    param_change = param_change.transpose()
    grad_change = grad_change.transpose()

    param_term1 = bk.dot(param_change)
    param_term1 = numpy.array(param_term1.dot(param_change.transpose()))
    param_term1 = param_term1.dot(bk)
    param_term2 = param_change.transpose().dot(bk)
    param_term2 = param_term2.dot(param_change)
    param_term = param_term1 / param_term2

    grad_term1 = grad_change.dot(grad_change.transpose())
    grad_term2 = grad_change.transpose().dot(param_change)
    grad_term = grad_term1 / grad_term2

    bk_next = bk - param_term + grad_term
    return bk_next
