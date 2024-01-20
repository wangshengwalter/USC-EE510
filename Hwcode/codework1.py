import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

plt.rcParams['figure.figsize']=8,8

def generate_and_plot(kx,mu):
    distr = multivariate_normal(
        cov = Kx, mean = mu,
        seed = 1000
    )
    data = distr.rvs(size = 5000)
    plt.grid()

    plt.plot(data[:,0],data[:,1],'o',c='line',markeredgewidth = 0.5, markeredgecolor = 'black')


    plt.title('Random samples from a 2D-Gaussain distribution')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis('equal')

Kx = np.array([[2.0, 1.0],[1.0, 4.0]])
mu = np.array([0,0])
random_seed = 10

generate_and_plot(Kx, mu)
