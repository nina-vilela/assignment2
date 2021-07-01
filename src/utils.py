from matplotlib import pyplot as plt
import numpy


def plot_contours_path(x0, x1, f):
    xx0, xx1 = numpy.meshgrid(numpy.linspace(-abs(max(x0)), abs(max(x0))),
                              numpy.linspace(-abs(max(x1)), abs(max(x1))))

    z = numpy.empty(shape=(xx0.shape[0], xx1.shape[1]))
    for i in range(xx0.shape[0]):
        for j in range(xx1.shape[1]):
            x = numpy.array([xx0[i][j], xx1[i][j]])
            z[i][j], _ = f(x)

    fig, ax = plt.subplots()
    cp = ax.contour(xx0, xx1, z)

    for i in range(len(x0) - 1):
        plt.annotate('', xy=(x0[i + 1], x1[i + 1]), xytext=(x0[i], x1[i]),
                     arrowprops={'arrowstyle': '-', 'color': 'r', 'lw': 0.3}, va='center', ha='center')

    ax.clabel(cp, inline=1, fontsize=10)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.scatter(x0, x1, s= 0.3)
    plt.show()


def new_plot(func_hist):
    its = list(range(len(func_hist)))
    plt.plot(its, func_hist)
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function')
    plt.show()
