import decision_tree as dt
import pandas as pd

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

