from utils import *
import math
import numpy as np
import matplotlib.pyplot as plt
import pprint


class MNB:

    def __init__(self):
        self.vocab = None
        self.class_prob = {}
        self.class_word_matrix = {}

    def lod_dict_prob(self, c, dataset):
        for instance in dataset:
            for word in instance:
                if word not in self.class_word_matrix[c]:
                    self.class_word_matrix[c][word] = 1
                else:
                    self.class_word_matrix[c][word] += 1

    def train(self, pos_train, neg_train, vocab):
        self.class_prob = {"+": len(pos_train) / (len(pos_train) + len(neg_train)),
                           "-": len(pos_train) / (len(pos_train) + len(neg_train))}
        self.class_word_matrix = {}
        self.vocab = vocab
        # pos_train_np = np.asarray([np.asarray(x) for x in pos_train],dtype=object)
        # print(np.unique(pos_train_np))
        for c in self.class_prob.keys():
            self.class_word_matrix[c] = {}
            if c == "+":
                self.lod_dict_prob(c, pos_train)
            elif c == "-":
                self.lod_dict_prob(c, neg_train)

    def display(self):
        print(self.class_prob)
        print()
        print(self.class_word_matrix)

    def predict(self, pos_test, neg_test):
        pos_prediction = []
        neg_prediction = []

        # 0 -negative, 1- positive

        pos_sum = 0
        for word in self.class_word_matrix["+"]:
            pos_sum += self.class_word_matrix["+"][word]
        neg_sum = 0
        for word in self.class_word_matrix["-"]:
            neg_sum += self.class_word_matrix["-"][word]

        for doc in pos_test:
            neg, pos = 1, 1
            for word in set(doc):
                if word in self.class_word_matrix["+"]:
                    pos_t = self.class_word_matrix["+"][word] / pos_sum
                else:
                    pos_t = 0

                if word in self.class_word_matrix["-"]:
                    neg_t = self.class_word_matrix["-"][word] / neg_sum
                else:
                    neg_t = 0
                neg *= neg_t
                pos *= pos_t

            neg = neg * self.class_prob["-"]
            pos = pos * self.class_prob["+"]
            pos_prediction.append((neg, pos))

        for doc in neg_test:
            neg, pos = 1, 1
            for word in set(doc):
                if word in self.class_word_matrix["+"]:
                    pos_t = self.class_word_matrix["+"][word] / pos_sum
                else:
                    pos_t = 0

                if word in self.class_word_matrix["-"]:
                    neg_t = self.class_word_matrix["-"][word] / neg_sum
                else:
                    neg_t = 0

                neg *= neg_t
                pos *= pos_t
            neg = neg * self.class_prob["-"]
            pos = pos * self.class_prob["+"]
            neg_prediction.append((neg, pos))

        pos_p = [max(range(len(j)), key=lambda i: j[i]) if j[0] != j[1] else random.randint(0, 1) for j in pos_prediction]
        neg_p = [max(range(len(j)), key=lambda i: j[i]) if j[0] != j[1] else random.randint(0, 1) for j in neg_prediction]
        return pos_p, neg_p

    def predict_log(self, pos_test, neg_test):
        pos_prediction = []
        neg_prediction = []

        epi = np.finfo(float).eps

        # 0 -negative, 1- positive

        pos_sum = 0
        for word in self.class_word_matrix["+"]:
            pos_sum += self.class_word_matrix["+"][word]
        neg_sum = 0
        for word in self.class_word_matrix["-"]:
            neg_sum += self.class_word_matrix["-"][word]

        for doc in pos_test:
            neg, pos = 0, 0
            for word in set(doc):
                if word in self.class_word_matrix["+"]:
                    pos_t = self.class_word_matrix["+"][word] / pos_sum
                else:
                    pos_t = 0

                if word in self.class_word_matrix["-"]:
                    neg_t = self.class_word_matrix["-"][word] / neg_sum
                else:
                    neg_t = 0
                neg += math.log(neg_t + epi)
                pos += math.log(pos_t + epi)

            neg = neg + math.log(self.class_prob["-"])
            pos = pos + math.log(self.class_prob["+"])
            pos_prediction.append((neg, pos))

        for doc in neg_test:
            neg, pos = 0, 0
            for word in set(doc):
                if word in self.class_word_matrix["+"]:
                    pos_t = self.class_word_matrix["+"][word] / pos_sum
                else:
                    pos_t = 0

                if word in self.class_word_matrix["-"]:
                    neg_t = self.class_word_matrix["-"][word] / neg_sum
                else:
                    neg_t = 0

                neg += math.log(neg_t + epi)
                pos += math.log(pos_t + epi)
            neg = neg + math.log(self.class_prob["-"])
            pos = pos + math.log(self.class_prob["+"])
            neg_prediction.append((neg, pos))

        pos_p = [max(range(len(j)), key=lambda i: j[i]) if j[0] != j[1] else random.randint(0, 1) for j in pos_prediction]
        neg_p = [max(range(len(j)), key=lambda i: j[i]) if j[0] != j[1] else random.randint(0, 1) for j in neg_prediction]

        return pos_p, neg_p

    def predict_laplace(self, pos_test, neg_test, alpha):
        pos_prediction = []
        neg_prediction = []

        # 0 -negative, 1- positive

        pos_sum = 0
        for word in self.class_word_matrix["+"]:
            pos_sum += self.class_word_matrix["+"][word]
        neg_sum = 0
        for word in self.class_word_matrix["-"]:
            neg_sum += self.class_word_matrix["-"][word]

        pos_sum += alpha * len(self.vocab)
        neg_sum += alpha * len(self.vocab)

        for doc in pos_test:
            neg, pos = 0, 0
            for word in set(doc):
                if word in self.class_word_matrix["+"]:
                    pos_t = (self.class_word_matrix["+"][word] + alpha) / pos_sum
                else:
                    pos_t = alpha/pos_sum

                if word in self.class_word_matrix["-"]:
                    neg_t = (self.class_word_matrix["-"][word] + alpha) / neg_sum
                else:
                    neg_t = alpha/neg_sum
                neg += math.log(neg_t)
                pos += math.log(pos_t)

            neg = neg + math.log(self.class_prob["-"])
            pos = pos + math.log(self.class_prob["+"])
            pos_prediction.append((neg, pos))

        for doc in neg_test:
            neg, pos = 0, 0
            for word in set(doc):
                if word in self.class_word_matrix["+"]:
                    pos_t = (self.class_word_matrix["+"][word] + alpha) / pos_sum
                else:
                    pos_t = alpha/pos_sum

                if word in self.class_word_matrix["-"]:
                    neg_t = (self.class_word_matrix["-"][word] + alpha) / neg_sum
                else:
                    neg_t = alpha/neg_sum

                neg += math.log(neg_t)
                pos += math.log(pos_t)
            neg = neg + math.log(self.class_prob["-"])
            pos = pos + math.log(self.class_prob["+"])
            neg_prediction.append((neg, pos))

        pos_p = [max(range(len(j)), key=lambda i: j[i]) if j[0] != j[1] else random.randint(0, 1) for j in
                 pos_prediction]
        neg_p = [max(range(len(j)), key=lambda i: j[i]) if j[0] != j[1] else random.randint(0, 1) for j in neg_prediction]
        return pos_p, neg_p

    def accuracy(self, conmat):
        n = sum(conmat["true+ve"].values()) + sum(conmat["true-ve"].values())

        return (conmat["true+ve"]["predicted+ve"] + conmat["true-ve"]["predicted-ve"]) / n

    def precision(self, conmat):
        n = conmat["true+ve"]["predicted+ve"] + conmat["true-ve"]["predicted+ve"]

        return conmat["true+ve"]["predicted+ve"] / n

    def recall(self, conmat):
        n = conmat["true+ve"]["predicted+ve"] + conmat["true+ve"]["predicted-ve"]

        return conmat["true+ve"]["predicted+ve"] / n

    def confusion_matrix(self, pos_p, neg_p):
        pos_p = np.asarray(pos_p)
        neg_p = np.asarray(neg_p)

        mat = {"true+ve": {"predicted+ve": 0, "predicted-ve": 0}, "true-ve": {"predicted+ve": 0, "predicted-ve": 0}}

        mat["true+ve"]["predicted+ve"] = np.sum(pos_p)
        mat["true+ve"]["predicted-ve"] = np.sum(np.sum(np.logical_not(pos_p)))
        mat["true-ve"]["predicted+ve"] = np.sum(neg_p)
        mat["true-ve"]["predicted-ve"] = np.sum(np.sum(np.logical_not(neg_p)))
        self.conmat = mat
        return mat


if __name__ == "__main__":
    MNB_classifier = MNB()

    # Q1
    percentage_positive_instances_train = 0.2
    percentage_negative_instances_train = 0.2

    percentage_positive_instances_test = 0.2
    percentage_negative_instances_test = 0.2
    # x = load_training_set(percentage_positive_instances_train,
    #                                         percentage_negative_instances_train)
    pos_tes, neg_test = load_test_set(percentage_positive_instances_test,
                                      percentage_negative_instances_test)
    # print(len(pos_tes),len(neg_test))
    MNB_classifier.train(*x)

    pos_tes, neg_test = load_test_set(percentage_positive_instances_test,
                                      percentage_negative_instances_test)
    simple = MNB_classifier.predict(pos_tes, neg_test)
    conmat = MNB_classifier.confusion_matrix(*simple)
    print(conmat)
    print(MNB_classifier.accuracy(conmat))
    print(MNB_classifier.precision(conmat))
    print(MNB_classifier.recall(conmat))

    log_p = MNB_classifier.predict_log(pos_tes, neg_test)
    conmat = MNB_classifier.confusion_matrix(*log_p)
    print(conmat)
    print(MNB_classifier.accuracy(conmat))
    print(MNB_classifier.precision(conmat))
    print(MNB_classifier.recall(conmat))

    # Q2
    percentage_positive_instances_train = 0.2
    percentage_negative_instances_train = 0.2

    percentage_positive_instances_test = 0.2
    percentage_negative_instances_test = 0.2

    MNB_classifier.train(*load_training_set(percentage_positive_instances_train,
                                            percentage_negative_instances_train))
    pos_tes, neg_test = load_test_set(percentage_positive_instances_test,
                                      percentage_negative_instances_test)
    conmat = MNB_classifier.confusion_matrix(*MNB_classifier.predict_laplace(pos_tes, neg_test, 1))
    print(conmat)
    print(MNB_classifier.accuracy(conmat))
    print(MNB_classifier.precision(conmat))
    print(MNB_classifier.recall(conmat))

    val = 0.0001
    X = []
    Y = []
    for i in range(7):
        X.append(val)

        conmat = MNB_classifier.confusion_matrix(*MNB_classifier.predict_laplace(pos_tes, neg_test, val))
        Y.append(MNB_classifier.accuracy(conmat))

        val *= 10
    plt.xscale("log")
    plt.plot(X,Y)
    plt.xlabel("Alpha")
    plt.ylabel("Accuracy")
    plt.title("Laplace smoothing")
    plt.show()

    best_alpha = 10
    # Q3
    percentage_positive_instances_train = 1
    percentage_negative_instances_train = 1

    percentage_positive_instances_test = 1
    percentage_negative_instances_test = 1

    MNB_classifier.train(*load_training_set(percentage_positive_instances_train,
                                            percentage_negative_instances_train))
    pos_tes, neg_test = load_test_set(percentage_positive_instances_test,
                                      percentage_negative_instances_test)
    log_p = MNB_classifier.predict_laplace(pos_tes, neg_test,best_alpha)
    conmat = MNB_classifier.confusion_matrix(*log_p)
    print(conmat)
    print(MNB_classifier.accuracy(conmat))
    print(MNB_classifier.precision(conmat))
    print(MNB_classifier.recall(conmat))

    # Q4
    percentage_positive_instances_train = 0.5
    percentage_negative_instances_train = 0.5

    percentage_positive_instances_test = 1
    percentage_negative_instances_test = 1

    MNB_classifier.train(*load_training_set(percentage_positive_instances_train,
                                            percentage_negative_instances_train))
    pos_tes, neg_test = load_test_set(percentage_positive_instances_test,
                                      percentage_negative_instances_test)
    log_p = MNB_classifier.predict_laplace(pos_tes, neg_test,best_alpha)
    conmat = MNB_classifier.confusion_matrix(*log_p)
    print(conmat)
    print(MNB_classifier.accuracy(conmat))
    print(MNB_classifier.precision(conmat))
    print(MNB_classifier.recall(conmat))

    # Q6
    percentage_positive_instances_train = 0.1
    percentage_negative_instances_train = 0.5

    percentage_positive_instances_test = 1
    percentage_negative_instances_test = 1

    MNB_classifier.train(*load_training_set(percentage_positive_instances_train,
                                            percentage_negative_instances_train))
    pos_tes, neg_test = load_test_set(percentage_positive_instances_test,
                                      percentage_negative_instances_test)
    log_p = MNB_classifier.predict_laplace(pos_tes, neg_test, best_alpha)
    conmat = MNB_classifier.confusion_matrix(*log_p)
    print(conmat)
    print(MNB_classifier.accuracy(conmat))
    print(MNB_classifier.precision(conmat))
    print(MNB_classifier.recall(conmat))
