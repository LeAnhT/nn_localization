import matplotlib.pyplot as plt
import numpy as np

def plot_ecdf(values,x_label,y_label):
    x = np.sort(values)
    y = np.arange(0,len(x))/len(x)
    plt.plot(x, y,'grey', marker='.', linestyle='none')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(np.arange(0,1,0.1))
    #plt.title(title)
    plt.show()

def simple_plot(values,x_label,y_label):
    plt.plot(np.arange(0,len(values)),values,'grey')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.set_cmap('Greys')
    plt.show()

def plot_grid_contour(x_size,y_size,step,grid_errors,x_label,y_label):
    X = np.arange(0, x_size + step, step)
    Y = np.arange(0, y_size + step, step)
    grid = np.array(grid_errors).reshape(len(X),len(Y))

    CS = plt.contour(X,Y,grid)
    for i in range(1,len(CS.collections)+1):
        CS.collections[i-1].set_label('line'+str(i))
    #plt.legend(loc='best')
    plt.clabel(CS,inline=1,fontsize=10)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.set_cmap('Greys')
    plt.clim(0,0.4)
    plt.show()
