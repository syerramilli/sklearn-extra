import numpy as np
import joblib

from skextra.pmf import PMF
from skextra.metrics import rmse
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

rmse_scorer = make_scorer(rmse, greater_is_better=False)

#######################################
# LOAD Data
#######################################
# Users and movies they have rated
X = np.loadtxt("movielens100K.data",np.uint16,delimiter="\t",usecols=[0,1])
# Corresponding ratings
y = np.loadtxt("movielens100K.data",np.float32,delimiter="\t",usecols=2)

# numbers counting from 1 onwards 
num_users = np.max(X[:,0])
num_items = np.max(X[:,1])

# recoding number from 0
X = X-1

#######################################
# Parameters and grid search over dimension
#######################################
pmf = PMF(lr=0.025,lambda_u=0.1,lambda_v=0.1,
          max_epoch=50,num_user=num_users,
          num_item =num_items,verbose=10)

param_grid = {'D':[2,5,10]}
model = GridSearchCV(pmf,param_grid,rmse_scorer,refit=True,cv=5,verbose=10)
model.fit(X,y)