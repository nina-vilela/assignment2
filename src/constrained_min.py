import numpy
from numpy import linalg as LA
import math
from src.unconstrained_min import newton_dir


def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):

    t = 1
    mu = 10
    tolerance = 10 ** (0 - 7)
    f_prev, grad_prev, hessian_prev = func(x0)
    phi = - math.log(numpy.prod(-f_prev))
    x_star = (t * f_prev) + phi

    m = 1 + len(ineq_constraints)
    termination_cond = m/t < tolerance
    x0_hist, x1_hist, f_hist = [x0[0]], [x0[1]], [f_prev]
    x_prev = x0

    while not termination_cond:
        t = mu * t
        print("f_prev: " + str(f_prev))
        print("grad_prev: " + str(grad_prev))

        #compute primal and dual Newton steps
        a = eq_constraints_mat
        pnt = newton_dir(grad_prev,hessian_prev)
        a_barrier = numpy.matrix([[hessian_prev, a.transpose()],[a, 0]])
        b_barrier = numpy.matrix([-grad_prev, 0])
        barrier_sol = LA.solve(a_barrier, b_barrier)

        #backtracking line search

        termination_cond = m/t < tolerance