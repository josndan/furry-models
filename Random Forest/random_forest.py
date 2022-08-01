from decision_tree import *
import numpy as np

class Random_Forest:
  """
  """
  def train(self, X, y, feature_type, criteria="IG",ntree = 10,stop_criteria=None):
    self.trees = []
    for i in range(ntree):
      X_1,y_1 = self.bag(X,y)
      # X_1,y_1 = X,y
      tree = Decision_Tree()
      tree.train(X_1,y_1,criteria,False,feature_type,True,stop_criteria)
      self.trees.append(tree)
  
  def bag(self,X,y):
    l = X.shape[0]
    per = np.random.randint(l,size = l)
    return X[per],y[per]

  def predict(self,X):
    prediction_lis = []
    for tree in self.trees:
      prediction_lis.append(tree.predict(X))
    array = np.stack(prediction_lis,axis = 1)
    res = []
    for row in array:
      val,count = np.unique(row,return_counts=True)
      i = np.argmax(count)
      res.append(val[i])
    return np.asarray(res)
