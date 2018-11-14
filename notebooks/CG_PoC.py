from scipy.sparse import csgraph
import numpy as np
import scipy

G = np.arange(7) * np.arange(7)[:, np.newaxis]
print (G)
GL = csgraph.laplacian(G, normed=False)
print (GL)

Gi1 = np.random.randint(-1, 2, size=(7, 7))
print (Gi1)

Gi2 = np.random.randint(-1, 2, size=(7, 7))
print (Gi2)