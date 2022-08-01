import pandas as pd
import numpy as np

def csv_to_numpy(filename,s):
    df = pd.read_csv(filename,sep=s)
    return df.to_numpy()

def get_wine_dataset():
    data = csv_to_numpy("./datasets/hw3_wine.csv","\t")
    return data[:,1:],data[:,0]

def get_house_votes_dataset():
    data = csv_to_numpy("./datasets/hw3_house_votes_84.csv",",")
    return data[:,:-1],data[:,-1]

def get_cancer_dataset():
    data = csv_to_numpy("./datasets/hw3_cancer.csv","\t")
    return data[:,:-1],data[:,-1]

def get_contraceptive_dataset():
    data = csv_to_numpy("./datasets/CMCD/cmc.data",",")
    return data[:,:-1],data[:,-1]

def load(name):
  if name=="wine":
    X, y = get_wine_dataset()
    f = np.zeros(X.shape[1])
    return X,y,f.astype(int)
  elif name == "house_votes":
    X,y = get_house_votes_dataset()
    f = np.ones(X.shape[1])
    return (X,y,f.astype(int))
  elif name == "cancer":
    X,y = get_cancer_dataset()
    f = np.zeros(X.shape[1])
    return (X,y,f.astype(int))
  elif name == "contraceptive":
    X,y = get_contraceptive_dataset()
    f = np.ones(X.shape[1])
    f[0] = 0
    f[3] = 0
    return (X,y,f.astype(int))