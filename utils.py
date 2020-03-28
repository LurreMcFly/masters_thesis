import os

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
    
    
def get_targets(x, df, ensemble, meta_model, num_targets=5, targets={}, greedy=False, anomaly='anomaly'):
    """
    Returns the true labels for the instances where 
    the model returns largest anomaly score.
    
    Args:
        x: numpy.array or torch.tensor of instances 
        df: pd.data_frame containing true labels
        ensemble: ensemble model
        meta_model : meta model
        target : dict containing previous targets
        greedy : true  - selecting targets greedy
                 false - selecting targets highest/lowest scores randomly 
        
    Returns:
        target : dict containing previous and new targets
    """
    
    if x.any() and ensemble and meta_model:
        #Get the total model output
        if isinstance(x, torch.Tensor):
            ens_scores = ensemble.get_anomaly_score(x.numpy())
            mm_outputs = meta_model(x)
        elif isinstance(x, np.ndarray):
            ens_scores = ensemble.get_anomaly_score(x)
            mm_outputs = meta_model(torch.Tensor(x))
        else:
            raise ValueError('x should be either torch.Tensor or numpy.ndarray')
    else:
        raise ValueError('Missing inputs')
        
    weighted_output = ens_scores * mm_outputs.detach().numpy()
    anomaly_scores = weighted_output.sum(axis=1)  
    
    if greedy:
        query_instances = np.array([x for x in np.argsort(anomaly_scores) if x not in targets][-num_targets:])
    else:
        if np.random.choice([0,1]):
            query_instances = np.array([x for x in np.argsort(anomaly_scores) if x not in targets][-num_targets:])
        else:
            query_instances = np.array([x for x in np.argsort(np.where(anomaly_scores > 0, anomaly_scores, np.inf))
                                       if x not in targets][:num_targets])
    true_labels = df.iloc[query_instances][anomaly].to_numpy()
    
    new_target = dict((int(idx), label) for  idx, label in zip(query_instances, true_labels))
    #appends new targets to old ones
    targets = {**targets, **new_target}
    
    return targets
    
    
def load_data_NAB(num, return_file_path=False):
    """
    Loads the NAB data set from folder 'data'
    
    Args:
        num: index of time series, should lie in 0-52(?)
    """
    
    paths = []
    for root, dirs, files in os.walk('data/NAB', topdown=False):
        for name in files:
            paths.append(os.path.join(root, name))
    
  
    #We only want the labeled data frames
    paths = [path for path in paths if path.split('_')[-1] == 'labeled.csv']

    file_path = paths[num]
    print(file_path)

    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    #df['anomaly'].replace(to_replace=0, value=-1, inplace=True)

    if return_file_path:
        return df, file_path
    else: 
        return df
    
def load_data_yahoo(num, return_file_path=False):
    """
    Loads the Yahoo data set from folder 'data'
    
    Args:
        num: index of time series, should lie in 0-52(?)
    """
    
    paths = []
    for root, dirs, files in os.walk('data/Yahoo', topdown=False):
        for name in files:
            paths.append(os.path.join(root, name))

    file_path = paths[num]
    print(file_path)

    df = pd.read_csv(file_path)
    #df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'is_anomaly' == df.columns[-1]:
        df['is_anomaly'].replace(to_replace=0, value=-1, inplace=True)
    else:
        df['anomaly'].replace(to_replace=0, value=-1, inplace=True)
        df.rename(columns={"timestamps": "timestamp"}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    if return_file_path:
        return df, file_path
    else: 
        return df
    
    
def train_test_split(df, p=0.4):
    """
    Splits df into train set and test set with p samples in train set
    
    Args:
        df : pd.data_frame to split
        p : fraction training set
    """
    train_df = df[:int(len(df)*p)]
    test_df = df[int(len(df)*p):]
    
    return train_df, test_df
    
    
def generate_data_window(df, window_size=11, lag0=6):
    """
    Converts data fram of time series values to 
    data frame with time series windows
    
    Args:
        window_size: size of time series window
        lag0: target index of window 
    """
    
    df_windows = df.copy()
    lag_neg = lag0 - 1
    lag_pos = window_size - lag0
    
    for x in range(-lag_neg, lag_pos + 1):
        df_windows['lag{}'.format(x)] = df_windows['value'].shift(x)
    
    #Special cases of slizing data frame
    if lag0 == 0:
        df_windows = df_windows.iloc[window_size:]
    elif lag0 == 1:
        df_windows = df_windows.iloc[window_size-1:-1]
    else:
        df_windows = df_windows.iloc[lag_pos:-lag_neg]
        
    return df_windows

def generate_sliding_window(df, window_size=10, anomaly_window=False):
    """
    Converts data fram of time series values to 
    data frame with time series windows
    Difference from generate_data_window:
        all windows covering the anomaly is seen as anomoulous
    
    Args:
        window_size: size of time series window
    """
    
    df_windows = df.copy()
    
    for x in range(0, window_size + 1):
        df_windows['lag{}'.format(x)] = df_windows['value'].shift(x)
    
    if anomaly_window:
        anomaly_idxs = df.loc[df['anomaly'] == 1].index
        window_idxs = [i+j for i in anomaly_idxs for j in range(window_size)]

        df_windows['anomaly'].iloc[window_idxs] = 1
        
    df_windows = df_windows.iloc[window_size:]
    
    return df_windows

def get_target_preds(ensemble, meta_model, x_targets):
    """
    Gets the fianl predictions
    
    Args:
        ensemble: ensemble model
        meta_model : meta model
        x_targets: numpy.array of instances
    
    Returns:
        target : torch.tensor of -1/1 (noraml/anomaly)
    """
    tensor_x_targets = torch.Tensor(x_targets)
    
    ens_scores = ensemble.get_anomaly_score(x_targets)
    mm_outputs = meta_model(tensor_x_targets)
        
    labeled_scores = (torch.tensor(ens_scores) * mm_outputs).sum(dim=1)
    labeled_preds = torch.ones(len(labeled_scores))
    labeled_preds[labeled_scores < 0] = -1
    
    return labeled_preds

def get_scores_preds(ensemble, meta_model, x_targets):
    """
    Gets the fianl anomaly scores
    
    Args:
        ensemble: ensemble model
        meta_model : meta model
        x_targets: numpy.array of instances
        
    Returns:
        target : numpy.array of final scores
    """
    tensor_x_targets = torch.Tensor(x_targets)
    
    ens_scores = ensemble.get_anomaly_score(x_targets)
    mm_outputs = meta_model(tensor_x_targets)
        
    labeled_scores = (torch.tensor(ens_scores) * mm_outputs).sum(dim=1)
    
    return labeled_scores.detach().numpy()

def my_loss(mm_outputs, target, score_x, score_x_prev, q_tau=1., lambda_prior=1.):
    """
    Prepares the loss to update the meta model
    
    Calculates the losses:
        Active Anomaly Detection loss: l_aad = max(0, y(q-score(x_t))) + max(0, y(score(x_t-1)-score(x_t)))
        Prior loss:                    l_prior = ensemble_prior * log(p) + (1-ensemble_prior) * log(p)
        Meta Model Loss:               l_mm = l_aad + lambda_prior * l_prior
        
    Args:
        mm_outputs:     output from meta model
        target:         labeled targets
        q_tau:          threshold
        score_x:        anomaly score at current time step
        score_x_prev:   anomaly score at previous time step
        ensemble_prior: prior for ensemble weights ex. [1/M or 1/sqrt(M)]
        lambda_prior:   constant for amount of weight on l_prior
    """
    if len(target):
        #number of assigned labels
        n_labels = len(target)

        # If n_labels == 0, no labels have been assigned -> l_add = 0
        y_ = torch.tensor(list(target.values()))
        
        l_aad = torch.mean(torch.max(torch.zeros(n_labels, dtype=float), y_ * (q_tau - score_x))) + \
                torch.mean(torch.max(torch.zeros(n_labels, dtype=float), y_ * (score_x_prev - score_x)))
        
    else:
        l_aad = torch.tensor([0])
        
    #calculates ensemble weighting prior
    ensemble_prior = torch.ones(mm_outputs.shape)/mm_outputs.shape[1]
    
    l_prior = F.binary_cross_entropy(mm_outputs, ensemble_prior, reduction='mean')
    
    l_mm = torch.sum(l_aad) + lambda_prior * torch.sum(l_prior)
    return l_mm