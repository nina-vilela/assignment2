from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
    fig1 = plt.figure()
    its = list(range(len(func_hist)))
    plt.plot(its, func_hist)
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function')
    plt.show()


def qp_plot(x_hist, f_hist, x_last):
    fig2 = plt.figure()
    ax = fig2.add_subplot(projection='3d')

    ax.plot(xs=[xxx[0] for xxx in x_hist], ys=[xxx[1] for xxx in x_hist], zs=[xxx[2] for xxx in x_hist], label="path")

    [x, y, z] = numpy.meshgrid(range(3), range(3), range(3))
    I = (x >= 0) & (y >= 0) & (z >= 0)
    ax.scatter(x[I], y[I], z[I], alpha=0.1)
    ax.scatter([1, 0, 0], [0, 1, 0], [0, 0, 1], color="pink")
    verts = list(zip([1, 0, 0], [0, 1, 0], [0, 0, 1]))
    ax.add_collection3d(Poly3DCollection([verts], color='pink', alpha=0.1))

    final_obj_val = f_hist[-1]
    ax.scatter(xs=x_last[0], ys=x_last[1], zs=x_last[2], alpha=1, color="red", label="final candidate, objective value =" + str(final_obj_val))

    plt.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


def lp_plot(x_hist, f_hist, x_last):
    xx = [x[0] for x in x_hist]
    yy = [x[1] for x in x_hist]

    # plot path taken by the algorithm
    plt.plot(xx, yy)

    d = numpy.linspace(-1, 2.5, 1000)
    x, y = numpy.meshgrid(d, d)
    I = (y >= -x + 1) & (y <= 1) & (x <= 2) & (y >= 0)
    plt.imshow((I).astype(int), extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greys", alpha=0.3);

    final_obj_val = int(f_hist[-1])
    plt.scatter(x=x_last[0], y=x_last[1], alpha=1, color="red",
               label="final candidate, objective value =" + str(final_obj_val))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

