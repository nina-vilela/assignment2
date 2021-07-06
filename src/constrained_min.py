import numpy
from numpy import linalg as LA
from functools import partial
from tests.examples import *
from src.unconstrained_min import newton_dir


def interior_pt(func, ineq_constraints_len, eq_constraints_mat, x0, eq_constraints_rhs=None):
    slope_ratio = math.exp(-4)
    back_track_factor = 0.2
    t = 1.0
    mu = 10.0
    tolerance = 10 ** (0 - 7)
    x_hist = []
    y_hist = []

    m = ineq_constraints_len
    termination_cond = m/t < tolerance
    while not termination_cond:
        cur_f = partial(f, func=func, t=t)
        y, grad, hess = cur_f(x=x0)
        x_hist.append(x0)
        y_hist.append(y)

        # if original problem has only inequality constraints, redirect to unconstrained Newton solver
        if eq_constraints_mat is None:
            direction = newton_dir(grad, hess)
        else:
            a_m, a_n = eq_constraints_mat.shape
            temp1 = numpy.append(hess, eq_constraints_mat.T, axis=1)
            temp2 = numpy.append(eq_constraints_mat, numpy.zeros((a_m, a_m)), axis=1)
            a = numpy.append(temp1, temp2, axis=0)
            b = numpy.append(-grad, numpy.zeros(a_m))
            direction = LA.solve(a, b)[:a_n]

        # backtracking
        step_size = 1.0
        a, _, _ = cur_f(x=(x0 + (step_size * direction)))
        b, _, _ = cur_f(x=x0)
        b += (slope_ratio * step_size * grad.transpose() * direction)
        sufficient_dec = (a <= b).all()
        while not sufficient_dec:
            step_size = back_track_factor * step_size
            a, _, _ = cur_f(x=(x0 + (step_size * direction)))
            b, _, _ = cur_f(x=x0)
            b += (slope_ratio * step_size * grad.transpose() * direction)
            sufficient_dec = (a <= b).all()

        x0 = x0 + (step_size * direction)
        t = mu * t
        termination_cond = m/t < tolerance
    return x_hist, y_hist, x0


def f(func, x, t):
    obj, y, grad, hess = func(x)
    phi_y = t * obj['y']
    phi_grad = t * obj['grad']
    phi_hess = t * obj['hess']

    for i in range(len(y)):
        phi_y += numpy.log(-y[i])
        phi_grad += grad[i] / (-y[i])
        m_grad = grad[i].shape[0]
        phi_hess += (grad[i].reshape(m_grad, 1) @ numpy.transpose(grad[i].reshape(m_grad, 1))) / (y[i]**2)
        phi_hess += hess[i] / (-y[i])
    return -phi_y, phi_grad, phi_hess



