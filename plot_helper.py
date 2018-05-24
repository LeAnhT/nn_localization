import matplotlib.pyplot as plt
import numpy as np

def plot_ecdf(values,x_label,y_label):
    x = np.sort(values)
    y = np.arange(0,len(x))/len(x)
    plt.plot(x, y, marker='.', linestyle='none')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.title(title)
    plt.show()

def simple_plot(values,x_label,y_label):
    plt.plot(np.arange(0,len(values)),values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def plot_grid_contour(x_size,y_size,step,grid_errors,x_label,y_label):
    X = np.arange(0, x_size + step, step)
    Y = np.arange(0, y_size + step, step)
    grid = np.array(grid_errors).reshape(len(X),len(Y))

    plt.contour(X,Y,grid)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
