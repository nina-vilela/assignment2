import unittest
from src.constrained_min import interior_pt
from tests.examples import *
from src.utils import *


class MyTestCase(unittest.TestCase):
    def test_qp(self):
        x0 = numpy.array([0.1, 0.2, 0.7])
        x_hist, f_hist, x_last = interior_pt(func=qp, ineq_constraints_len=3,
                                             eq_constraints_mat=numpy.array([[1, 1, 1]]), x0=x0)
        qp_plot(x_hist, f_hist, x_last)

    def test_lp(self):
        x0 = numpy.array([0.5, 0.75])
        x_hist, f_hist, x_last = interior_pt(func=lp, ineq_constraints_len=4, eq_constraints_mat=None, x0=x0)
        lp_plot(x_hist, f_hist, x_last)


if __name__ == '__main__':
    unittest.main()
