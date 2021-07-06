import numpy
import math


def f1_1(x, hessian_flag=False):
    y = x[0]**2 + x[1]**2
    gradient = 2 * x
    if hessian_flag:
        return y, gradient, numpy.array([[2, 0], [0, 2]])
    return y, gradient


def f1_2(x, hessian_flag=False):
    y = (5 * (x[0]**2)) + x[1]**2
    gradient = numpy.array([10*x[0], 2*x[1]])
    if hessian_flag:
        return y, gradient, numpy.array([[10, 0], [0, 2]])
    return y, gradient


def f1_3(x, hessian_flag=False):
    y = (4*(x[0]**2)) - ((2*math.sqrt(3))*x[0]*x[1]) + 2*(x[1]**2)
    gradient = numpy.array([8*x[0], 4*x[1]])
    if hessian_flag:
        return y, gradient, numpy.array([[8, 0], [0, 4]])
    return y, gradient


def f2(x, hessian_flag=False):
    y = 100*(x[1] - (x[0])**2)**2 + (1 - x[0])**2

    x0 = -400 * (x[1] - x[0]**2)*x[0] - 2*(1 - x[0])
    x1 = 200 * (x[1] - x[0]**2)
    gradient = numpy.array([x0, x1])
    if hessian_flag:
        hessian = numpy.array([[1200*(x[0]**2) - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200]])
        return y, gradient, hessian
    return y, gradient


def f3(x, hessian_flag=False):
    y = 2*x[0] + 2*x[1]
    return y, numpy.array([2])


def qp(x):
    y = numpy.array(x[0]**2 + x[1]**2 + (x[2] + 1)**2)
    grad = numpy.array([2 * x[0], 2 * x[1], 2 * (x[2]+1)])
    hess = numpy.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    obj = {'y':y, 'grad':grad, 'hess':hess}

    y1 = -x[0]
    grad1 = numpy.array([-1, 0, 0])
    hess = numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    y2 = -x[1]
    grad2 = numpy.array([0, -1, 0])

    y3 = -x[2]
    grad3 = numpy.array([0, 0, -1])

    return obj, numpy.array([y1, y2, y3]), numpy.array([grad1, grad2, grad3]), numpy.array([hess, hess, hess])


def lp(x):
    y = -x[0] - x[1]
    grad = numpy.array([-1, -1])
    hess = numpy.zeros((2, 2))
    obj = {'y': y, 'grad': grad, 'hess': hess}

    y1 = -x[0] - x[1] + 1
    grad1 = numpy.array([-1, -1])

    y2 = x[1] - 1
    grad2 = numpy.array([0, 1])

    y3 = x[0] - 2
    grad3 = numpy.array([1, 0])

    y4 = -x[1]
    grad4 = numpy.array([0, -1])

    return obj, numpy.array([y1, y2, y3, y4]), numpy.array([grad1, grad2, grad3, grad4]), \
        numpy.array([hess, hess, hess, hess])

