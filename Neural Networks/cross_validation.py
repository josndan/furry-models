import numpy as np
from random import choices, sample


class CrossValidation:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k

    def get_folds(self):
        tot = self.y.shape[0]
        vals = np.unique(self.y)
        cat = []
        ratio = []
        for val in vals:
            n = self.y[self.y == val].shape[0]
            cat.append(np.nonzero(self.y == val)[0])
            ratio.append(n / tot)
        ratio = np.asarray(ratio)
        ratio[-1] = 1 - (ratio.sum() - ratio[-1])
        n_each_fold = tot // self.k
        fold_index = []
        fold_X = []
        fold_y = []
        for i in range(self.k):
            startified = []
            strat_X = None
            strat_y = np.asarray([], dtype=np.dtype(int))
            joined_in = np.asarray([], dtype=np.dtype(int))
            n_ele = 0
            for j, r in enumerate(ratio):
                temp = np.random.choice(cat[j], min(int(r * n_each_fold),len(cat[j])),replace=False)
                n_ele += temp.shape[0]
                startified.append(temp)
                cat[j] = np.setdiff1d(cat[j], temp)
                if strat_X is None:
                    strat_X = self.X[temp]
                else:
                    strat_X = np.concatenate((strat_X, self.X[temp]), axis=0)
                strat_y = np.concatenate((strat_y, self.y[temp]), axis=0)
                joined_in = np.concatenate((joined_in, cat[j]), axis=0)
            if n_ele < n_each_fold:
                temp = np.random.choice(joined_in, int(n_each_fold - n_ele),replace=False)
                startified.append(temp)
                for j in range(len(cat)):
                    cat[j] = np.setdiff1d(cat[j], temp)
                strat_X = np.concatenate((strat_X, self.X[temp]), axis=0)
                strat_y = np.concatenate((strat_y, self.y[temp]), axis=0)

            fold_index.append(startified)
            fold_X.append(strat_X)
            fold_y.append(strat_y)
        result = []
        for i in range(self.k):
            trains_X = fold_X[:i] + fold_X[i + 1:]
            trains_y = fold_y[:i] + fold_y[i + 1:]

            train_X = np.concatenate(trains_X, axis=0)
            train_y = np.concatenate(trains_y, axis=0)

            train = (train_X, train_y)
            test = (fold_X[i], fold_y[i])
            result.append((train, test))

        return result

    def recall_two_class(self, y_predict, y):
        one = np.ones(y.shape)
        total_true_pos = y_predict[one == y].shape[0]
        total_pred_pos = np.sum(y_predict[one == y])

        if total_true_pos==0:
          return float("inf")
        return total_pred_pos / total_true_pos

    def precision_two_class(self, y_predict, y):
        one = np.ones(y_predict.shape)
        total_pred_pos = y[one == y_predict].shape[0]
        total_true_pos = np.sum(y[one == y_predict])
        if total_pred_pos ==0:
          return float("inf")
        return total_true_pos / total_pred_pos

    def make_pos(self, y, val):
        divided = y.copy()
        pos_ind = divided == val
        neg_ind = divided != val
        divided[pos_ind] = 1
        divided[neg_ind] = 0
        return divided

    def multi_class(self, y_predict, y, metric):
        vals = np.unique(y)
        res = []
        if vals.shape[0] == 2:
            return metric(y_predict, y)
        else:
            for val in vals:
                temp = metric(self.make_pos(y_predict, val), self.make_pos(y, val))
                res.append(temp)
        return np.ma.masked_invalid(np.asarray(res)).mean()

    def accuracy(self, y_predict, y):
        return np.sum(y_predict == y)/y.shape[0]

    def recall(self, y_predict, y):
        return self.multi_class(y_predict, y, self.recall_two_class)

    def precision(self, y_predict, y):
        return self.multi_class(y_predict, y, self.precision_two_class)
    
    def f1_score(self,y_predict,y):
      p =self.precision(y_predict,y)
      a = self.accuracy(y_predict,y)
      if a == float("inf") or p == float("inf"):
        return float("nan")
      return 2 *(p*a)/(p+a)

