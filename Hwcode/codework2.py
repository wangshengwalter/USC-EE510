import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

x = np.loadtxt("gmm_mixture.csv", delimiter=",", dtype=float)


components = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
likelihood_dataset = np.zeros(9)

for i in range(2, 11):
    gmm = GaussianMixture(n_components=i, n_init=10)
    gmm.fit(x)
    print("means")
    means = gmm.means_
    print(means)
    print("covariances")
    covariances = gmm.covariances_
    print(covariances)
    print("weights")
    weights = gmm.weights_
    print(gmm.weights_)

    # likelihood_dataset = 1.0
    # for dataitem in x:
    #     likelihood_item = 0.0000000000000
    #     for group in range(0, i):
    #         # likelihood_item += weights[group] * multivariate_normal.pdf(dataitem, mean=means[group], cov=covariances[group])
    #         # likelihood_item += weights[group] * (1/(((2*np.pi)**(len(dataitem)/2))*np.sqrt(np.linalg.det(covariances[group]))))*np.exp(-0.5*np.dot(np.dot((dataitem-means[group]).T, np.linalg.inv(covariances[group])), (dataitem-means[group])))
    #     likelihood_dataset *= likelihood_item


    likelihood_dataset[i-2] = np.exp(gmm.score(x))
    print("likelihood when J = " + str(i))
    print("%.8f" % likelihood_dataset[i-2])


plt.grid()
plt.plot(components,likelihood_dataset)
plt.xlabel('J')
plt.ylabel('likelihood')
plt.show()







