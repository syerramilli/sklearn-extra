import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class PMF(BaseEstimator, ClassifierMixin):
    def __init__(self, D=10, lr=1, lambda_u=0.1, lambda_v=0.1, 
                 max_epoch=20, batch_size=128, num_user=1, num_item=1,
                 tol = 1e-6, red_factor= 0.8,
                 verbose=False,random_state=None):
        self.__name__ = "PMF"
        self.D = D # latent features
        self.lr = lr
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v

        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        
        self.tol = tol
        self.red_factor = red_factor

        if random_state is None:
            self.rng = np.random.RandomState(np.random.randint(0,2e+4))
        else:
            self.rng = random_state
        
        self.verbose = verbose

        self.U = None # user vectors
        self.V = None # product vectors


    def fit(self, X, y):
        """A reference implementation of a fitting function
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """

        n_samples = X.shape[0]
        
        self.U =  0.1 * self.rng.randn(self.num_user, self.D) # user vectors
        self.V =  0.1 * self.rng.randn(self.num_item, self.D) # item vectors

        mask_u = np.zeros((self.num_user, self.D))
        mask_v = np.zeros((self.num_item, self.D))
        self.mean_rating_ = np.mean(y)
        self.min_rating_ = np.min(y)
        self.max_rating_ = np.max(y)
        
        lr = self.lr*1
        
        for epoch in range(self.max_epoch):
            shuffled_ids = np.arange(X.shape[0])
            self.rng.shuffle(shuffled_ids)
            
            for i in range(0,n_samples,self.batch_size):
                batch_size = min(self.batch_size,n_samples-i)
                batch_user_ids = X[shuffled_ids[i:i+batch_size],0]
                batch_item_ids = X[shuffled_ids[i:i+batch_size],1]
                ratings = y[shuffled_ids[i:i+batch_size]] - self.mean_rating_ 
                
                # predictions and error
                pred_out = np.sum(np.multiply(self.U[batch_user_ids, :], 
                                              self.V[batch_item_ids, :]), 
                                    axis=1)
                error = pred_out-ratings
                
                # compute gradients
                grad_u = np.multiply(error[:, np.newaxis], 
                                     self.V[batch_item_ids, :]) +\
                                     self.lambda_u * self.U[batch_user_ids, :]
                grad_v = np.multiply(error[:, np.newaxis],
                                     self.U[batch_user_ids, :]) +\
                                     self.lambda_v * self.V[batch_item_ids, :]
                
                # update parameters with masking
                mask_u = np.zeros((self.num_user, self.D))
                mask_v = np.zeros((self.num_item, self.D))
                for t,user_id in enumerate(batch_user_ids):
                    mask_u[user_id] = grad_u[t]

                for t,item_id in enumerate(batch_item_ids):
                    mask_v[item_id] = grad_v[t]

                mask_u = np.clip(mask_u, -3, 3)
                mask_v = np.clip(mask_v, -3, 3)

                self.U = self.U - lr * mask_u
                self.V = self.V - lr * mask_v

            #lr = lr * self.red_factor
            
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
            Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        
        n_samples = X.shape[0]
        y_pred = self.mean_rating_*np.ones(n_samples)
        for i in np.arange(n_samples):
            if X[i,0] < self.num_user and X[i,1] < self.num_item:
                y_pred[i] += np.dot(self.U[X[i,0],:],self.V[X[i,1],:])
                y_pred[i] = np.clip(y_pred[i],self.min_rating_,
                      self.max_rating_)
                
        return y_pred