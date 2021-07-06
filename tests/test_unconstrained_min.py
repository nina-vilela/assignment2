import unittest
from tests.examples import *
from src.unconstrained_min import *
from src.utils import *


class MyTestCase(unittest.TestCase):
    def test_quad_min(self):
        quad_functions = [f1_1, f1_2, f1_3]
        for cur_ex in quad_functions:
            last_loc, success, x0, x1, f_hist = line_search(cur_ex, numpy.array([1, 1]), 0.1, 10**(-8),  10**(-12), 100, 'nt')
            plot_contours_path(x0, x1, cur_ex)
            new_plot(f_hist)
        self.assertEqual(success, True)

    def test_rosenbrock_min(self):
        last_loc, success, x0, x1, f_hist = line_search(f2, numpy.array([2, 2]), 0.001, 10**(-7), 10 **(-8), 10000, 'nt')
        plot_contours_path(x0, x1, f2)
        new_plot(f_hist)
        self.assertEqual(success, True)

    def test_lin_min(self):
        last_loc, success, x0, x1, f_hist = line_search(f3, numpy.array([2, 2]), 0.001, 10**(-7), 10**(-8), 600)
        plot_contours_path(x0, x1, f3)
        self.assertEqual(success, True)

if __name__ == '__main__':
    unittest.main()