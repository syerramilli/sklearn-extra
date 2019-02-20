import numpy as np
import time
import copy
from sklearn.base import BaseEstimator, ClassifierMixin

class PMF(BaseEstimator, ClassifierMixin):
    def __init__(self, D=10, lr=1, lambda_u=0.1, lambda_v=0.1, 
                 max_epoch=20, batch_size=1000, num_user=1, num_item=1,
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

        #self.rmse_train = []
        #self.rmse_test = []

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

        start_time = time.time()

        num_train = X.shape[0]

        epoch = 0
        num_batches = num_train // self.batch_size

        if num_train % self.batch_size != 0:
            num_batches += 1  

        self.U = 0.1 * np.random.randn(self.num_user, self.D) # user vectors
        self.V = 0.1 * np.random.randn(self.num_item, self.D) # product vectors

        mask_u = np.zeros((self.num_user, self.D))
        mask_v = np.zeros((self.num_item, self.D))
        #self.mean_rating_ = np.mean(y)

        best_rmse = 10000
        waiting = 0
        
        lr = copy.copy(self.lr)
        
        while epoch < self.max_epoch:
            epoch += 1

            shuffled_ids = np.arange(X.shape[0])
            np.random.shuffle(shuffled_ids)

            predictions = np.array([])
            golds = np.array([])

            for i in range(num_batches):   
                batch_ids = np.arange(self.batch_size * i, min(num_train, self.batch_size * (i + 1)))
                batch_size = len(batch_ids)

                batch_user_ids = np.array(X[shuffled_ids[batch_ids], 0], dtype='int32')
                batch_item_ids = np.array(X[shuffled_ids[batch_ids], 1], dtype='int32')
                batch_golds = np.array(y[shuffled_ids[batch_ids]], dtype='int32')

                # compute objective function
                pred = np.sum(np.multiply(self.U[batch_user_ids, :], self.V[batch_item_ids, :]), axis=1)                
                predictions = np.append(predictions, pred)
                golds = np.append(golds, batch_golds)

                # calculate gradients
                error = pred - batch_golds #+ self.mean_rating_
                grad_u = np.multiply(error[:, np.newaxis], self.V[batch_item_ids, :]) + self.lambda_u * self.U[batch_user_ids, :]
                grad_v = np.multiply(error[:, np.newaxis], self.U[batch_user_ids, :]) + self.lambda_v * self.V[batch_item_ids, :]

                # print(grad_u)
                # print(">", grad_v)

                # update parameters with masking
                t = 0
                for user_id in batch_user_ids:
                    mask_u[user_id] = grad_u[t]
                    t+=1

                t = 0
                for item_id in batch_item_ids:
                    mask_v[item_id] = grad_v[t]
                    t+=1

                mask_u = np.clip(mask_u, -3, 3)
                mask_v = np.clip(mask_v, -3, 3)

                self.U = self.U - lr * mask_u
                self.V = self.V - lr * mask_v

            lr = lr * self.red_factor
            
            error = predictions -golds#+ self.mean_rating_ - golds
            rmse = np.linalg.norm(error) / np.sqrt(num_train)

            if epoch % 10 == 0 and self.verbose:
                print("Epoch:",epoch, "rmse:", rmse)

            if np.isnan(rmse):
                if self.verbose:
                    print("early stop - nan")
                    print("best train rmse:", best_rmse, "for D=",self.D, 
                          "lr=", self.lr, "lambda_u=", self.lambda_u, 
                          "lambda_v=", self.lambda_v, "train time=", 
                          time.time()-start_time, "total epoch=", epoch)
                break

            if abs(best_rmse - rmse) > 1e-06 :
                best_rmse = rmse
                waiting = 0
            else:
                waiting+=1

            if waiting >= 2 or epoch == self.max_epoch:
                if self.verbose:
                    print("best train rmse:", best_rmse, "for D=",self.D, 
                          "lr=", self.lr, "lambda_u=", self.lambda_u, 
                          "lambda_v=", self.lambda_v, "train time=", 
                          time.time()-start_time, "total epoch=", epoch)
                break
                
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

        num_test = len(X)
        eval_num_batches = num_test // self.batch_size
        if num_test % self.batch_size != 0:
            eval_num_batches += 1

        predictions = np.array([])

        for i in range(eval_num_batches):
            batch_ids = np.arange(self.batch_size * i, min(num_test, self.batch_size * (i + 1)))

            batch_user_ids = np.array(X[[batch_ids], 0], dtype='int32')
            batch_item_ids = np.array(X[[batch_ids], 1], dtype='int32')

            # print(np.multiply(self.U[batch_user_ids, :], self.V[batch_item_ids, :]).shape)
            pred = np.sum(np.multiply(self.U[batch_user_ids, :], self.V[batch_item_ids, :]), axis=2).squeeze()
            predictions = np.append(predictions, pred)
            
        return predictions#+self.mean_rating_