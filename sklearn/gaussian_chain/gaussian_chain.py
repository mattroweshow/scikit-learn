__author__ = 'rowem'

import numpy as np
import threading
from scipy.stats import norm
import Queue

class Gaussian:
    def __init__(self, mean, variance):
        """Fit Gaussian Chain classifier according to X, y

        Parameters
        ----------
        mean : Mean of the Gaussian, calculated via maximum likelihood

        variance : variance of the Gaussian, calculated via maximum likelihood

        Returns
        -------
        self : object
            Returns self.
        """
        self.mean = mean
        self.variance = variance

    def prob_membership(self, value):
        gaussian_dist = norm(self.mean, self.variance)
        prob = gaussian_dist.pdf(value)
        # print self
        # print value
        return prob

    def __str__(self):
        return "N(" + str(self.mean) + ", " + str(self.variance) + ")"

class SingleGaussianChain():

    def __init__(self, rho, alpha, lambdaA, eta, learning_mode, target_class):
        """
        The Single Binary Gaussian Chain model

        Parameters
        ___________

        rho : {the smoothing parameter for zero-probabilities in the model.
            These are thrown when a feature value's probability of featuring the Gaussian is minimal

        """
        self.rho = rho
        self.alpha = alpha
        self.lambdaA = lambdaA
        self.eta = eta
        self.learning_mode = learning_mode
        self.target_class = target_class


    def fit(self, X, y):
        """Fit Gaussian Chain classifier according to X, y

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns self.
        """

        # Prime the model's Gaussians
        # print "Priming the model's Gaussians"
        self.prime(X, y)

        # 1. Initialise the learning setting
        e = 0
        self.b = [0] * X.shape[1]
        # Save the prior epoch parameters
        self.b_old = [1] * X.shape[1]

        # # Prime the classes info
        class_set = set(y)
        self.classes = range(0, max(class_set)+1)


        # 2. Run learning until the model converges between epochs or we max out
        # print "Learning the model"
        # Learn the model till convergence
        while self.b != self.b_old and e < 100:
            # print "Learning Epoch: " + str(e)
            # Reset the parameter vector
            self.b_old = self.b

            # Learn the model based on the learning routine
            # single stochastic learning routine
            if self.learning_mode == 1:
                # Shuffle the instance indices ordering of X
                np.random.seed(0)
                instance_indices = np.random.permutation(X.shape[0])
                for i in instance_indices:
                    # print "Instance Index: " + str(i)
                    # Determine the churn probability
                    churn_prob = self.derive_prob(X, i)

                    # Determine the error
                    error_i = y[i] - churn_prob
                    # print "Error: " + str(error_i)

                    # Update the model's parameters based on the error derivative using elastic net regularisation
                    if error_i != 0:
                        for j in range(0, len(self.b)-1, 1):
                            if self.checkGaussian(j):
                                # Determine the first order derivative of the objective
                                delta_ij = -1 * X[i,j] * (error_i + self.lambdaA * (1 - self.alpha) * self.b[j] + self.lambdaA * self.alpha)

                                # Update the jth parameter based on the derivative
                                self.b[j] = self.b[j] - self.eta * delta_ij
                                # print self.b

            # dual stochastic learning routine
            elif self.learning_mode == 2:
                # Shuffle the instance indices ordering of X
                np.random.seed(0)
                instance_indices = np.random.permutation(X.shape[0])
                for i in instance_indices:
                    # print "Instance Index: " + str(i)
                    # shuffle the parameter vector elements
                    feature_indices = np.random.permutation(X.shape[1])
                    for j in feature_indices:
                        # print str(j)
                        if self.checkGaussian(j):
                            # Determine the churn probability
                            churn_prob = self.derive_prob(X, i)
                            # Determine the error
                            error_i = y[i] - churn_prob
                            # print "Error: " + str(error_i)
                            if error_i != 0:
                                # Update the model's parameters based on the error derivative using elastic net regularisation
                                # Determine the first order derivative of the objective
                                delta_ij = -1 * X[i,j] * (error_i + self.lambdaA * (1 - self.alpha) * self.b[j] + self.lambdaA * self.alpha)

                                # Update the jth parameter based on the derivative
                                self.b[j] = self.b[j] - self.eta * delta_ij

            # Increment the learning epoch
            e+=1

        # print "Convered after epoch: " + str(e)
        # print self.b


    def prime(self, X, y):
        print "Priming Gaussians"

        # For each attribute in X, determine the maximum likelihood mean and variance
        # Single case so only consider classes of one label - specified by the target class in the constructor
        # Get index indices of the target class
        # target_indices = [i for i in range(0, len(y)-1, 1) if y[i] == self.target_class]
        self.gaussians = {}

        # Prime the threads
        self.exitFlag = 0
        threadList = range(0, X.shape[0])
        self.queueLock = threading.Lock()
        self.workQueue = Queue.Queue(X.shape[0])
        self.computedQueue = {}
        self.threads = []

        # Create new threads
        for tName in threadList:
            # Set the model type to be 1 (as single Gaussian Sequence model)
            thread = gcThread(self, 1, tName,  self.workQueue, self.computedQueue, X, y)
            thread.start()
            self.threads.append(thread)

        # Fill the queue
        self.queueLock.acquire()
        for j in range(0, X.shape[1], 1):
            # Call prime thread
            self.workQueue.put(j)
        self.queueLock.release()

        # Wait for queue to empty
        while not self.workQueue.empty():
            pass

        # Notify threads it's time to exit
        self.exitFlag = 1

        # Wait for all threads to complete
        for t in self.threads:
            t.join()

        # Add the computed Gaussians into the Gaussian Chain Model
        for c in self.computedQueue.keys():
            # print str(c) + " -> " + str(self.computedQueue.get(c))
            self.gaussians[c] = self.computedQueue.get(c)

    def prime_gaussian(self, q, c_q, X, y):
        while not self.exitFlag:
            self.queueLock.acquire()
            if not self.workQueue.empty():
                id = q.get()
                # print "From queue = " + str(id)
                # compute gaussian from data
                j_array = []
                target_indices = [i for i in range(0, len(y)-1, 1) if y[i] == self.target_class]
                for i in target_indices:
                    j_array.append(X[i, id])
                mean = np.mean(j_array)
                var = np.var(j_array)
                c_q[id] = Gaussian(mean, var)
                # print str(id) + " has Gaussian: " + str(c_q[id])
                self.queueLock.release()
                # print str(self.workQueue.qsize())
                # print "%s processing %s" % (threadName, data)
            else:
                self.queueLock.release()

    # Checks if the Gaussian at the index can be used (i.e. doesn't have 0 mean and var)
    def checkGaussian(self, j):
        mean = self.gaussians[j].mean
        variance = self.gaussians[j].variance

        if mean != 0 and variance != 0:
            return True
        else:
            return False

    # Predicts the churn probability given the instance vector
    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples]
            Returns the probability of the sample for the target class
        """
        print "Applying Gaussians to induce probabilities"

        # Get all churn probs
        churn_probs = []

        for i in range(0,X.shape[0]):
            churn_prob = self.derive_prob(X, i)
            churn_probs.append(churn_prob)

        # Normalise churn probs
        normaliser = max(churn_probs)
        # norm_churn_probs = churn_probs / normaliser

        # Work out class specific probabilities
        probabilities = []
        # For binary classes : n x 2 matrix
        for i in range(0, X.shape[0]):
            churn_prob = self.derive_prob(X, i) / normaliser
            probabilities.append(churn_prob)

        # print probabilities
        return probabilities

    def derive_prob(self, X, i):
        churn_prob = 1
        for j in range(0, len(self.b)-1, 1):
            if self.checkGaussian(j):
                # print str(j)
                gaussian_prob = self.gaussians[j].prob_membership(X[i, j])
                # print "probability of membership = " +  str(gaussian_prob)
                b_gaussian_prob = self.b[j] * gaussian_prob

                if b_gaussian_prob <= 0:
                    b_gaussian_prob = self.rho

                # Take the joint probability calculation
                churn_prob *= b_gaussian_prob
        return churn_prob

class DoubleGaussianChain():

    def __init__(self, rho, alpha, lambdaA, eta, learning_mode, target_class):
        """
        The Single Binary Gaussian Chain model

        Parameters
        ___________

        rho : {the smoothing parameter for zero-probabilities in the model.
            These are thrown when a feature value's probability of featuring the Gaussian is minimal

        """
        self.rho = rho
        self.alpha = alpha
        self.lambdaA = lambdaA
        self.eta = eta
        self.learning_mode = learning_mode
        self.target_class = target_class


    def fit(self, X, y):
        """Fit Gaussian Chain classifier according to X, y

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns self.
        """

        # Prime the model's Gaussians
        # print "Priming the model's Gaussians"
        self.prime(X, y)

        # 1. Initialise the learning setting
        e = 0
        self.b = [0] * X.shape[1]
        # Save the prior epoch parameters
        self.b_old = [1] * X.shape[1]

        # # Prime the classes info
        class_set = set(y)
        self.classes = range(0, max(class_set)+1)

        # 2. Run learning until the model converges between epochs or we max out
        # print "Learning the model"
        # Learn the model till convergence
        while self.b != self.b_old and e < 100:
            # print "Learning Epoch: " + str(e)
            # Reset the parameter vector
            self.b_old = self.b

            # Learn the model based on the learning routine
            # single stochastic learning routine
            if self.learning_mode == 1:
                # Shuffle the instance indices ordering of X
                np.random.seed(0)
                instance_indices = np.random.permutation(X.shape[0])
                for i in instance_indices:
                    # print "Instance Index: " + str(i)
                    # Determine the churn probability
                    churn_prob = self.derive_prob(X, i)

                    # Determine the error
                    error_i = y[i] - churn_prob
                    # print "Error: " + str(error_i)

                    # Update the model's parameters based on the error derivative using elastic net regularisation
                    if error_i != 0:
                        for j in range(0, len(self.b)-1, 1):
                            if self.checkPosGaussian(j) or self.checkNegGaussian(j):

                                # Determine the first order derivative of the objective
                                delta_ij = -1 * X[i, j] * (error_i + self.lambdaA * (1 - self.alpha) * self.b[j] + self.lambdaA * self.alpha)

                                # Update the jth parameter based on the derivative
                                self.b[j] = self.b[j] - self.eta * delta_ij
                                # print self.b

            # dual stochastic learning routine
            elif self.learning_mode == 2:
                # Shuffle the instance indices ordering of X
                np.random.seed(0)
                instance_indices = np.random.permutation(X.shape[0])
                for i in instance_indices:
                    # print "Instance Index: " + str(i)
                    # shuffle the parameter vector elements
                    feature_indices = np.random.permutation(X.shape[1])
                    for j in feature_indices:
                        # print str(j)
                        if self.checkPosGaussian(j) or self.checkNegGaussian(j):
                            # Determine the churn probability
                            churn_prob = self.derive_prob(X, i)
                            # Determine the error
                            error_i = y[i] - churn_prob
                            # print "Error: " + str(error_i)
                            if error_i != 0:
                                # Update the model's parameters based on the error derivative using elastic net regularisation
                                # Determine the first order derivative of the objective
                                delta_ij = -1 * X[i,j] * (error_i + self.lambdaA * (1 - self.alpha) * self.b[j] + self.lambdaA * self.alpha)

                                # Update the jth parameter based on the derivative
                                self.b[j] = self.b[j] - self.eta * delta_ij

            # Increment the learning epoch
            e+=1

        # print "Convered after epoch: " + str(e)
        # print self.b


    def prime(self, X, y):
        print "Priming Gaussians"

        # For each attribute in X, determine the maximum likelihood mean and variance
        # Single case so only consider classes of one label - specified by the target class in the constructor
        # Get index indices of the target class
        self.pos_gaussians = {}
        self.neg_gaussians = {}

        # Prime the threads
        self.exitFlag = 0
        threadList = range(0, X.shape[1])
        self.queueLock = threading.Lock()
        self.workQueue = Queue.Queue(X.shape[0])
        self.computedQueue = {}
        self.threads = []

        # Create new threads
        for tName in threadList:
            # Set the model type to be 2 (as double Gaussian Sequence model)
            thread = gcThread(self, 2, tName, self.workQueue, self.computedQueue, X, y)
            thread.start()
            self.threads.append(thread)

        # Fill the queue
        self.queueLock.acquire()
        for j in range(0, X.shape[1], 1):
            # Call prime thread
            self.workQueue.put(str(j) + "_pos")
            self.workQueue.put(str(j) + "_neg")
        self.queueLock.release()

        # Wait for queue to empty
        while not self.workQueue.empty():
            pass

        # Notify threads it's time to exit
        self.exitFlag = 1

        # Wait for all threads to complete
        for t in self.threads:
            t.join()

        # Add the computed Gaussians into the Gaussian Chain Model: as positive and negative gaussians
        for c in self.computedQueue.keys():
            if "_pos" in c:
                c_new = c.replace("_pos", "")
                self.pos_gaussians[int(c_new)] = self.computedQueue.get(c)
            else:
                c_new = c.replace("_neg", "")
                self.neg_gaussians[int(c_new)] = self.computedQueue.get(c)

    def prime_gaussian(self, q, c_q, X, y):
        while not self.exitFlag:
            self.queueLock.acquire()
            if not self.workQueue.empty():
                id = q.get()
                old_id = id
                # print "From queue = " + str(id)
                # compute gaussian from data
                j_array = []
                target_indices = []
                if "_pos" in id:
                    target_indices = [i for i in range(0, len(y)-1, 1) if y[i] == self.target_class]
                    id = id.replace("_pos", "")
                else:
                    target_indices = [i for i in range(0, len(y)-1, 1) if y[i] != self.target_class]
                    id = id.replace("_neg", "")

                for i in target_indices:
                    j_array.append(X[i, int(id)])

                mean = np.mean(j_array)
                var = np.var(j_array)
                c_q[old_id] = Gaussian(mean, var)
                self.queueLock.release()
            else:
                self.queueLock.release()

    # Checks if the Gaussian at the index can be used (i.e. doesn't have 0 mean and var)
    def checkPosGaussian(self, j):
        mean = self.pos_gaussians[j].mean
        variance = self.pos_gaussians[j].variance

        if mean != 0 and variance != 0:
            return True
        else:
            return False

    def checkNegGaussian(self, j):
        mean = self.neg_gaussians[j].mean
        variance = self.neg_gaussians[j].variance

        if mean != 0 and variance != 0:
            return True
        else:
            return False

    # Predicts the churn probability given the instance vector
    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples]
            Returns the probability of the sample for the target class
        """
        print "Applying Gaussians to induce probabilities"

        # Get all churn probs
        churn_probs = []

        for i in range(0,X.shape[0]):
            churn_prob = self.derive_prob(X, i)
            churn_probs.append(churn_prob)

        # Normalise churn probs
        normaliser = max(churn_probs)
        # norm_churn_probs = churn_probs / normaliser

        # Work out class specific probabilities
        probabilities = []
        # For binary classes : n x 2 matrix
        for i in range(0, X.shape[0]):
            churn_prob = self.derive_prob(X, i) / normaliser
            probabilities.append(churn_prob)

        # print probabilities
        return probabilities

    def derive_prob(self, X, i):
        pos_prob = 1
        for j in range(0, len(self.b)-1, 1):
            # Get the probability of the feature value being in the positive gaussian
            pos_gaussian_prob = 0
            if self.checkPosGaussian(j):
                # print str(j)
                pos_gaussian_prob = self.pos_gaussians[j].prob_membership(X[i, j])

            # Get the probability of the feature value being in the negative gaussian
            neg_gaussian_prob = 0
            if self.checkNegGaussian(j):
                neg_gaussian_prob = self.neg_gaussians[j].prob_membership(X[i, j])
                # print "probability of membership = " +  str(gaussian_prob)

            b_gaussian_prob = self.b[j] * pos_gaussian_prob - (1 - self.b[j]) * neg_gaussian_prob
            if b_gaussian_prob <= 0:
                b_gaussian_prob = self.rho

            # Take the joint probability calculation
            pos_prob *= b_gaussian_prob
        return pos_prob

# Class to handle parallelised Gaussian induction
class gcThread (threading.Thread):
    def __init__(self, gaussian_chain_model, model_type, threadID, q, c_q, X, y):
        threading.Thread.__init__(self)
        self.gaussian_chain_model = gaussian_chain_model
        self.model_type = model_type
        self.threadID = threadID
        self.q = q
        self.c_q = c_q
        self.X = X
        self.y = y

    def run(self):
        # print "Starting " + str(self.gaussian_id)
        if self.model_type is 1:
            SingleGaussianChain.prime_gaussian(self.gaussian_chain_model,
                                           self.q, self.c_q, self.X, self.y)
        else:
            DoubleGaussianChain.prime_gaussian(self.gaussian_chain_model,
                                           self.q, self.c_q, self.X, self.y)
        # print "Exiting " + str(self.gaussian_id)




