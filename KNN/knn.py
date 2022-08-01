import numpy as np
from collections import Counter

class KNN:
  def train(self, X, y):
    
    f_max = np.amax(X,axis=0)
    f_min = np.amin(X,axis=0)

    self.X_train = X - f_min / (f_max - f_min)
    self.y_train = y
  
  def perdict(self,X,k):

    dist = np.sqrt(np.sum((self.X_train ** 2),axis=1) + np.sum((X ** 2),axis = 1)[:,None] - 2 * X.dot(self.X_train.T)) 
    y_predict = np.ones(dist.shape[0])
    for i in range(dist.shape[0]):
      ind = np.argsort(dist[i])[:k]
      neighb_y = self.y_train[ind]
      neighb_count = Counter(neighb_y)
      y_predict[i] = neighb_count.most_common(1)[0][0]
    
    return y_predict

