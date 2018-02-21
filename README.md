# OMGP!

Oh My GP! is a Julia package in development for **extremely efficient Gaussian Processes algorithms**. It contains for the moment only two classifiers : the Bayesian SVM, and a state-of-the-art algorithm for classification using the logit link called X-GPC. It is planned to implement Regression, as well as more complex likelihood, including the multi-class classification.

## Install the package

Run in Julia `Pkg.clone("git://github.com/theogf/OMGP.jl.git")`, it will install the package and all its requirements


## Use the package

A complete documentation is currently being written, for now you can use this very basic example where `X_train` is a matrix ``N x D`` where `N` is the number of training points and `D` is the number of dimensions and `Y_train` is a vector of outputs.

```
using OMGPC
model = SparseXGPC(X_train,Y_train;Stochastic=true,BatchSize=100,m=64,Kernels=[Kernel["rbf",1.0,params=1.0]]) #Parameters after ; are optional
model.train(iterations=100)
Y_predic = model.predict(X_test) #For getting the label directly
Y_predic_prob = model.predictproba(X_test) #For getting the likelihood of predicting class 1
```

There is also a more complete example in a Julia notebook : [Classification with Sparse XGPC][31b06e91]

## References :

["Gaussian Processes for Machine Learning"](http://www.gaussianprocess.org/gpml/) by Carl Edward Rasmussen and Christopher K.I. Williams

ECML 17' "Bayesian Nonlinear Support Vector Machines for Big Data" by Florian Wenzel, Théo Galy-Fajou, Matthäus Deutsch and Marius Kloft. [https://arxiv.org/abs/1707.05532][arxivbsvm]

Arxiv "Efficient Gaussian Process Classification using Polya-Gamma Variables" [https://arxiv.org/abs/1802.06383][arxivxgpc]

  [31b06e91]: https://github.com/theogf/OMGP.jl/blob/master/examples/Classification%20-%20SXGPC.ipynb "Classification with Sparse XGPC"
[arxivbsvm]:https://arxiv.org/abs/1707.05532
[arxivxgpc]:https://arxiv.org/abs/1802.06383
