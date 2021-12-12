import pandas as pd
import numpy as np
import statistics

# this code calls tensorflow 1 compatibility
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import seaborn as sns
import DeepAE as DAE
import l21shrink as SHR  # also changing to l1 to access function
import xlrd
import numpy.linalg as nplin
import matplotlib.pyplot as plt

class RDAE(object):
    """
    @author: Chong Zhou
    2.0 version.
    complete: 10/17/2016
    version changes: move implementation from theano to tensorflow.
    3.0
    complete: 2/12/2018
    changes: delete unused parameter, move shrink function to other file
    update: 03/15/2019
        update to python3
    Des:
        X = L + S
        L is a non-linearly low rank matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_1
        Use Alternating projection to train model
    """
    def __init__(self, sess, layers_sizes, lambda_=1.0, error = 1.0e-7):
        """
        sess: a Tensorflow tf.Session object
        layers_sizes: a list that contain the deep ae layer sizes, including the input layer
        lambda_: tuning the weight of l1 penalty of S
        error: converge criterior for jump out training iteration
        """
        self.lambda_ = lambda_
        self.layers_sizes = layers_sizes
        self.error = error
        self.errors=[]

        # made changes to deep AE so encoder only takes 1 row as input, will likely break this..
        self.AE = DAE.Deep_Autoencoder( sess = sess, input_dim_list = self.layers_sizes)

    def fit(self, X, sess, learning_rate=0.15, inner_iteration = 50,
            iteration=1, batch_size=1, verbose=False):

        ## The first layer must be the input layer, so they should have same sizes.
        assert type(X) == np.matrix, "Data must be a numpy matrix."
        assert X.shape[1] == self.layers_sizes[0], "The number of columns should match the first layer."

        ## initialize L, S, mu(shrinkage operator)
        ### could try setting array of zeros equal to the number of columns... # changing from X.shape to X.shape[1]
        self.L = np.zeros(X.shape[1])
        self.S = np.zeros(X.shape[1])

        print("L shape:", self.L.shape) # L is 141 x 20; need to treat each row as it's own element.

        print("X size is:", X.size, "(in case it requires adjusting too)")

        # X.size: .size is a np function that counts all elements in a matrix (np array) along the specified axis.
        # if no axis is specified, counts all elements in the matrix.
        mu = (X.size) / (4.0 * nplin.norm(X,1))
        print ("shrink parameter:", self.lambda_ / mu)
        LS0 = self.L + self.S

        XFnorm = nplin.norm(X,'fro')
        if verbose:
            print ("X shape: ", X.shape)
            print ("L shape: ", self.L.shape)
            print ("S shape: ", self.S.shape)
            print ("mu: ", mu)
            print ("XFnorm: ", XFnorm)

        for it in range(iteration):
            if verbose:
                print ("Out iteration: " , it)
            ## alternating project, first project to L
            self.L = X - self.S

            ## Using L to train the auto-encoder
            self.AE.fit(X = self.L, sess = sess,
                                    iteration = inner_iteration,
                                    learning_rate = learning_rate,
                                    batch_size = batch_size,
                                    verbose = verbose)

            ## get optmized L
            self.L = self.AE.getRecon(X = self.L, sess = sess)

            ## alternating project, now project to S
            ### *changed SHR.shrink to l21shrink * ###
            ### * should we be using l21 shrink? ### --> check paper...
            self.S = SHR.l21shrink(self.lambda_/mu, (X - self.L).reshape(X.size)).reshape(X.shape)

            ## break criterion 1: the L and S are close enough to X
            c1 = nplin.norm(X - self.L - self.S, 'fro') / XFnorm

            ## break criterion 2: there is no changes for L and S
            c2 = np.min([mu,np.sqrt(mu)]) * nplin.norm(LS0 - self.L - self.S) / XFnorm

            if verbose:
                print ("Break Criterion 1: the L and S are close enough to X", c1)
                print ("Break Criterion 2: there is no changes for L and S", c2)

            if c1 < self.error and c2 < self.error :
                print ("early break")
                break

            ## save L + S for c2 check in the next iteration
            LS0 = self.L + self.S

        return self.L , self.S

    def transform(self, X, sess):
        L = X - self.S
        return self.AE.transform(X = L, sess = sess)

    def getRecon(self, X, sess):
        return self.AE.getRecon(X, sess = sess)


if __name__ == "__main__":
    df = pd.read_csv("Leukemia_GSE22529_U133B.csv")
    X = df.iloc[:,2:].to_numpy()

    X = np.matrix(X.T)
    m,n = X.shape

    with tf.Session() as sess:
        layers = [n, int(n * 0.5)]
        rae = RDAE(sess=sess, lambda_=30, layers_sizes=layers)
        # rae = RDAE(sess=sess, lambda_=2000, layers_sizes=[784, 400])

        L, S = rae.fit(X, sess=sess, learning_rate=0.01, batch_size=40, inner_iteration=50,
                       iteration=5, verbose=True)

        l21R = rae.getRecon(X, sess=sess)
        l21H = rae.transform(X, sess)

        print("cost errors, not used for now:", rae.errors)

    fig, axs = plt.subplots(5, 5, figsize=(15, 15))
    count = 0
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].plot(l21H[:, count])
            count += 1



