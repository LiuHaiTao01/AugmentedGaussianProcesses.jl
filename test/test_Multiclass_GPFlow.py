import gpflow
import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
plt.style.use('ggplot')

#make a one dimensional classification problem
data = np.genfromtxt("data/artificial-characters_train")
X = data[:,1:].copy()
Y = data[:,0].copy()-1
test_data = np.genfromtxt("data/artificial-characters_train")
X_test = test_data[:,1:].copy()
Y_test = test_data[:,0].copy()-1

## labels needs to be in the format 0,1,2,3,4 etc

ind_points = KMeans(n_clusters=100).fit(X).cluster_centers_
m = gpflow.models.SVGP(X, Y, kern=gpflow.kernels.RBF(7) + gpflow.kernels.White(7, variance=0.01),likelihood=gpflow.likelihoods.MultiClass(10), num_latent=10, Z=ind_points)

m.kern.kernels[1].variance.trainable = False
m.feature.trainable = False

opt = gpflow.train.AdamOptimizer()
opt.minimize(m,maxiter=1000)
p, _ = m.predict_y(X_test)
best = np.array([np.argmax(p[i,:]) for i in range(Y_test.size)])
score = 0
for i in range(best.size):
    if best[i]==Y_test[i]:
        score += 1

score /= Y_test.size
print('Accuracy is {}', score)
