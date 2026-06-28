import pandas as pd
import numpy as np
from pathlib import Path

from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomLinkSplit
from sklearn.model_selection import train_test_split

def load_dataset(path, sep=" ", ascending=True, casting="str"):

  assert Path(path).is_file() == True

  edges = pd.DataFrame()
  if sep == ' ':
    df = pd.read_csv(
        path,
        sep=sep,
        header=None,
        names=["source", "target", "time"],
        usecols=["source", "target", "time"],
    )

    if casting == "str":
      edges[['source', 'target']] = df[['source', 'target']].astype(str)
    elif casting =="int":
      edges[["source", "target"]] = df[["source", "target"]].astype(int)
    
    edges['time'] = df.time.values
  elif sep == ',':

    df = pd.read_csv(path)
    # ,u,i,ts,label,idx

    df[["u", "i"]] = df[["u", "i"]].astype(str)
    #df[['ts', 'label']] = df[['ts', 'label']].astype(int)
    df[['ts']] = df[['ts']].astype(int)
   

    edges['source'] = df.u.values
    edges['target'] = df.i.values
    edges['time'] = df.ts.values

  edges.sort_values(by='time', ascending=ascending)

  return edges


def train_val_test_split(data, negative_sampling=False, device='cpu'):
       
    transform = T.Compose([
                    T.NormalizeFeatures(),
                    T.ToDevice(device),
                    T.RandomLinkSplit(num_val=0.1,
                                num_test=0.20,
                                is_undirected=True,
                                add_negative_train_samples=negative_sampling,
                                neg_sampling_ratio=1.0)

                ])
    
    transform = T.RandomLinkSplit(num_val=0.10, 
                              num_test=0.20,
                              is_undirected=True,
                              add_negative_train_samples=True,
                              neg_sampling_ratio=1.0)
    
    train_data, val_data, test_data = transform(data)

    return train_data, val_data, test_data

