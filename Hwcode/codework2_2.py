import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

x = np.loadtxt("pca_data.csv", delimiter=",", dtype=float)

pca = PCA(n_components = 200)
pca.fit(x)

plt.figure(1, figsize=(8, 6))
plt.grid()
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('Explained_variance_')
plt.show()