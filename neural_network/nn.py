import random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier

# From the forum https://groups.google.com/forum/#!topic/ml17pd/c77ll68gFSE

mnist = fetch_mldata("MNIST original")
# rescale the data, use the traditional train/test split
X, y = mnist.data / 255., mnist.target
print "# of samples " + str(y.size)
# print X[:, 1].size

N_tr = 20000
N_te = 10000
print "# of training " + str(N_tr)
print "# of test " + str(N_te)

# idx contiene tutti gli indici presi casualmente da un range che va da 0 a 70000(che e il massimo)
# ne prende quanti necessari per il training e per il test
idx = random.sample(range(70000), N_tr+N_te)


X_train, X_test = X[idx[:N_tr]], X[idx[N_tr:N_tr+N_te]]
y_train, y_test = y[idx[:N_tr]], y[idx[N_tr:N_tr+N_te]]


# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)

# Inizializzo una rete Multi Layer Perceptron
# con un solo strato da 100 neuroni
# 100 iterazioni di apprendimento
# alpha: L2 penalty (regularization term) parameter. (sembra un fattore di "dimenticanza")
# errata corrige, questo parametro viene utilizzato in http://scikit-learn.org/stable/modules/neural_networks_supervised.html#regularization
# in MLPRegressor per evitare overfitting
# solver non ne so un cazzo
# tol fattore di tolleranza, se la rete non migliora le sue prestazioni di almeno 2*tol allora termina l'apprendimento
# learning rate inti: sarebbe 'eta'
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

# apprendimento
mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

# la loss e la funzione che calcola l errore complessivo

fig, axes = plt.subplots(5, 5)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    #ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin, vmax=.5 * vmax)
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
