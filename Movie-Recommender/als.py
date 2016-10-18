################################################################
# Author: Ankoor Bhagat                                        #                                           
# Date: 10/06/2016                                             #                             
# ALS Collaborative Filtering (Implicit feedback data)         #
# Version: 1.0 (not optimized for speed)                       #
################################################################

# Imports
import numpy as np


# Alternating Least Squares Class (Implicit data)
class ALS(object):
    """
    Alternating Least Squares Collaborative Filtering for Implicit Feedback Data
    
    Implicit Feedback: Data (like clicks, etc.) is proxy of user preference

    Algorithm based on: "Collaborative Filtering for Implicit Feedback Datasets" paper
                         by Yifan Hu, Yehuda Koren, Chris Volinsky
    
    Paper source: http://yifanhu.net/PUB/cf.pdf 
    """

    def __init__(self, R, f=5, lambda_=0.5, n_iter=50, alpha=10):
        """
        Inputs:
            R: user-item ranking matrix (size: m x n), where m = # of users, n = # of items
            f: # of features
            lambda_: regularization parameter
            n_iter: # of iterations
            alpha: rate of increase in confudence
        """
        # Inputs
        self.R = R
        self.f = f
        self.lambda_ = lambda_
        self.n_iter = n_iter
        self.alpha = alpha

        # Derived inputs
        self.m = R.shape[0] # number of users
        self.n = R.shape[1] # number of items

        # Preference and Confidence when user-factor fixed
        self.P = (self.R > 0).astype(int) # User preference (implicit)
        self.C = 1 + self.alpha * self.R # Our confidence

        # # Preference and Confidence when item-factor fixed
        # self.PT = self.P.T
        # self.CT = self.C.T
    
    def train(self):
        """
        Train model using Alternating Least Squares algorithm
        """
        # 0. Initialize user-factor matrix X (size: m x f)
        self.X = np.random.rand(self.m, self.f) * 0.1
        
        # 1. Initialize item-factor matrix Y (size: n x f)
        self.Y = np.random.rand(self.n, self.f) * 0.1

        for iteration in xrange(self.n_iter):

            # 2. Fix item-factor matrix and solve for user-factor matrix
            self.X = self.alternating_least_squares(self.Y, self.X, self.C, self.P)

            # 3. Fix user-factor matrix (use transpose of C and P) and solve for item-factor matrix
            self.Y = self.alternating_least_squares(self.X, self.Y, self.C.T, self.P.T)

            # Print
            if (iteration % 5) == 0:
                print 'Iteration: {} finished'.format(iteration)


    def alternating_least_squares(self, fixed, update, C_, P_):
        """
        Alternating Least Squares step
        Inputs: 
            fixed: user-factor or item-factor matrix
            update: item-factor or user-factor matrix
            C_: Confidence matrix (use transpose when user-factor matrix is fixed)
            P_: User preference matrix (use transpose when user-factor matrix is fixed)
        Output: Updated user-factor or item-factor matrix
        """
        X = update
        Y = fixed
        # Compute: Y'Y matrix of size f x f
        YT_Y = Y.T.dot(Y) 
        # For each user 
        for u in xrange(X.shape[0]): 
            # Define diagonal matrix C^u of size n x n
            Cu = np.diag(C_[u,:]) 
            # Define vector p(u) of size n
            pu = P_[u,:] 
            # Compute: Y'.C^u.p(u)
            YT_Cu_pu = Y.T.dot(Cu).dot(pu)
            # Compute: Y'.C^u.Y
            YT_Cu_Y = YT_Y + Y.T.dot(Cu - np.eye(Cu.shape[0])).dot(Y)
            # Compute: (Y'.C^u.Y + lambda_.I)^-1
            inv = np.linalg.inv(YT_Cu_Y + self.lambda_ * np.eye(YT_Cu_Y.shape[0]))
            # Compute: Xu
            X_u = inv.dot(YT_Cu_pu)
            # update user-factor
            X[u,:] = X_u 
        return X
    
    def predict(self):
        """
        P_hat = XT.Y
        """
        P_hat = self.X.dot(self.Y.T)
        return P_hat

    def mean_squared_error(self):
        """
        Calculate mean squared error
        """
        P_hat = self.predict()
        mse = ((self.P.flatten() - P_hat.flatten())**2).mean() 
        return mse

