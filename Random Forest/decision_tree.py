import numpy as np
import pandas as pd
from random import choices, sample
import math


class Node:
    def __init__(self):
        self.test_feature = None
        self.children = []
        self.keys = []
        self.label = None

    def print(self, space=0):
        print(space * "  " + f"Test feature: {self.test_feature}, label: {self.label}")
        for child in self.children:
            child.print(space + 1)

    def print_myself(self):
        print(f"Test feature: {self.test_feature}, label: {self.label} ")


class Decision_Tree:
    def train(self, X, y, criteria, remove_features=True, feature_type=None, random=False, stop_criteria=None):
        self.X_train = X
        self.y_train = y
        self.root = self.create_tree_helper(
            criteria, stop_criteria, remove_features, feature_type, random
        )

    def default_stop(self, root, available_features, X_train, y_train, fun,max_criteria):
        y_unique, y_count = np.unique(y_train, return_counts=True)
        if y_unique.shape[0] == 1:
            root.label = y_unique[0]
            return True
        
        res = True
        for i in range(X_train.shape[1]):
          x_unique = np.unique(X_train[:,i])
          if x_unique.shape[0] != 1:
            res = False

        if len(available_features) < 1:
            root.label = y_unique[np.argmax(y_count)]
            return True

        return False or res

    def minimal_size_for_split_stop_helper(self, root, available_features, X_train, y_train, fun,max_criteria,n):
        if y_train.shape[0] < n or self.default_stop(root, available_features, X_train, y_train, fun,max_criteria):
            return True

        return False
    
    def minimal_size_for_split_stop(self,n):
      return lambda r,a,x,y,fun,max_criteria: self.minimal_size_for_split_stop_helper(r,a,x,y,fun,max_criteria,n)

    def minimal_gain_criterion_helper(self, root, available_features, X_train, y_train, fun,max_criteria,threshold):
        if max_criteria < threshold or self.default_stop(root, available_features, X_train, y_train, fun,max_criteria):
            return True

        return False

    def minimal_gain_criterion(self,threshold):
      return lambda r,a,x,y,fun,max_criteria: self.minimal_size_for_split_stop_helper(r,a,x,y,fun,max_criteria,threshold)

    def create_tree_helper(self, criteria, stop_criteria, remove_features, feature_type, random):
        if feature_type is None:
            feature_type = np.ones(self.X_train.shape[1])
        self.feature_type = feature_type
        root = Node()
        num_features = self.X_train.shape[1]
        fun = None
        if stop_criteria is None:
          stop_criteria = self.default_stop
        if criteria == "IG":
            fun = self.information_gain
        elif criteria == "Gini":
            fun = lambda y, x: -self.compute_information(x, self.gini_coefficient)
        self.create_tree(
            root,
            set(range(num_features)),
            self.X_train,
            self.y_train,
            fun,
            stop_criteria,
            remove_features,
            feature_type,
            random,
        )
        return root

    def create_tree(
        self,
        root,
        available_features,
        X_train,
        y_train,
        evaluator,
        stopping_criteria,
        remove_features,
        feature_type,
        random,
        max_criteria = float("-inf")
    ):
        y_unique, y_count = np.unique(y_train, return_counts=True)
        # print("y_train",y_train)
        # if y_count.size != 0:
        # print(y_count)
        root.label = y_unique[np.argmax(y_count)]
        # else:
          # root.label = 
        if stopping_criteria(root, available_features, X_train, y_train, evaluator,max_criteria):
            return
        # print("new")
        df = pd.DataFrame(X_train)
        df["true"] = y_train
        # print(df)
        max_i = -1
        max_criteria = float("-inf")
        s = available_features
        best_mean = None
        best_val, best_cat = None, None
        cs = []
        
        res = True
        while random and res:
            s = sample(
                list(available_features), int(math.sqrt(len(available_features)))
            )
            for i in s:
              x_unique = np.unique(X_train[:,i])
              if x_unique.shape[0] != 1:
                res = False
        for i in s:
            vals, category = None, None
            if feature_type[i]:
                vals, category = self.split_categorical(X_train[:, i])
            else:
                best_mean, category = self.split_numerical(
                    X_train[:, i], y_train, evaluator
                )
            split_y = [y_train[indices] for indices in category]
            criteria = evaluator([y_train], split_y)
            cs.append((s,criteria))
            dont = False
            if criteria == max_criteria:
              for q in category:
                if q.size == 0:
                  dont = True
        
            if criteria > max_criteria or (criteria == max_criteria and not dont):
                max_criteria = criteria
                max_i = i
                best_val = vals
                best_cat = category
        # print("max_i",max_i)
        # print("Criterias",cs)
        for q in best_cat:
          if q.size == 0:
            print(df)
            print(best_cat)
            print(best_mean)
            self.split_numerical(X_train[:, i], y_train, evaluator,True)
            print(cs)
            print(i)
            raise Exception
        root.test_feature = max_i
        if remove_features:
            available_features.remove(max_i)

        if not feature_type[max_i]:
            # v, c = self.split(self.X_train[:,max_i])
            best_val = [best_mean, best_mean]
        
        for i,cat in enumerate(best_cat):
            node = Node()
            # if i < len(category):
            val = best_val[i]
            self.create_tree(
                node,
                set(available_features),
                X_train[cat],
                y_train[cat],
                evaluator,
                stopping_criteria,
                remove_features,
                feature_type,
                random,
                max_criteria
            )
            # else:
            #   y_unique,y_count = np.unique(self.y_train,return_counts=True)
            #   node.label = y_unique[np.argmax(y_count)]

            root.children.append(node)
            root.keys.append(val)

    def predict(self, X, debug=False):
        y_predict = []

        for X_i in X:
            y_predict.append(self.predict_recursive(X_i, self.root, debug, 0))

        return np.asarray(y_predict)
        
    def predict_recursive(self, X_i, root, debug, space):
        if len(root.children) == 0:
            # if debug:
            #     root.print_myself()
            #     print(space * "  " "0 children")
            #     space = 1
            return root.label

        # if debug:
        #     root.print_myself()
        next_node = None
        if self.feature_type[root.test_feature]:
            i = int(X_i[root.test_feature])
            if i in root.keys:
                next_node = root.children[root.keys.index(i)]
            else:
                # if debug:
                #     space = 1
                #     print(space * "  " "Feature missing the test data value")
                return root.label
        else:
            i = int(X_i[root.test_feature])
            mean = root.keys[0]
            if i <= mean:
                next_node = root.children[0]
            else:
                next_node = root.children[1]
        return self.predict_recursive(X_i, next_node, debug, space)


    def information_gain(self, parent, child):
        return self.compute_information(
            parent, self.entropy
        ) - self.compute_information(child, self.entropy)

    # Category is a list of list where each element in category is
    # the list of values from each split
    def compute_information(self, category, fun):
        w_i = []
        tot = 0
        e_i = []
        for val in category:
            w_i.append(val.shape[0])
            e_i.append(fun(val))
            tot += val.shape[0]

        w_i = np.asarray(w_i)
        e_i = np.asarray(e_i)
        w_i = w_i / tot

        return np.sum(w_i * e_i)

    def split_categorical(self, data):
        vals = np.unique(data)
        category = []
        for val in vals:
            category.append(np.nonzero(data == val)[0])
        return vals, category

    def split_numerical(self, data, y_train, evaluator,p=False):
        df = pd.DataFrame({"numerical_feature": data.ravel()})
        df = df.sort_values("numerical_feature")
        df["rolling_mean"] = df.rolling(2).mean()
        max_split_val = -1
        max_cat = None
        max_criteria = float("-inf")
        best_mean = 0
        if p :
          print(df)
        for index, row in df[1:].iterrows():
            category = []
            category.append(
                df[df.numerical_feature <= row["rolling_mean"]].index.values
            )
            category.append(df[df.numerical_feature > row["rolling_mean"]].index.values)
            split_y = [y_train[indices] for indices in category]
            criteria = evaluator([y_train], split_y)
            if p:
              print(criteria)
            if criteria > max_criteria:
                max_criteria = criteria
                max_split_val = row["rolling_mean"]
                max_cat = category
                best_mean = row["rolling_mean"]

        return best_mean, max_cat

    def entropy(self, data):
        vals, category = self.split_categorical(data)
        p_i = []
        tot = 0
        for val in category:
            p_i.append(val.shape[0])
            tot += val.shape[0]
        p_i = np.asarray(p_i)
        p_i = p_i / tot
        return -np.sum(p_i * np.log2(p_i))

    def gini_coefficient(self, data):
        vals, category = self.split(data)
        p_i = []
        tot = 0
        for val in category:
            p_i.append(val.shape[0])
            tot += val.shape[0]
        p_i = np.asarray(p_i)
        p_i = p_i / tot
        return 1 - np.sum(np.power(p_i, 2))
