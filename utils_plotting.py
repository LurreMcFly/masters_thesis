import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

from IPython import display
import time


def plot_anomaly_series(df, x=None, ensemble=None, meta_model=None, targets=None, column='lag0'):
    """
    Plots the results of the series in df
    
    Args:
        df: pd.data_frame with columns 'timestamp', 'lag0', 'anomaly', ('window')
        x: numpy.array or torch.tensor of instances
        ensemble: ensemble model
        meta_model : meta model
        targets : dict containing previous targets
        column : column in containing actual series
    """
    
    fig, ax = plt.subplots(figsize=(16,4))
    
    if isinstance(x, (np.ndarray, torch.Tensor)) and ensemble and meta_model:
        #Get the total model output
        if isinstance(x, torch.Tensor):
            ens_scores = ensemble.get_anomaly_score(x.numpy())
            mm_outputs = meta_model(x)
        elif isinstance(x, np.ndarray):
            ens_scores = ensemble.get_anomaly_score(x)
            mm_outputs = meta_model(torch.Tensor(x))
        else:
            raise ValueError('x should be either torch.Tensor or numpy.ndarray')

        weighted_output = ens_scores * mm_outputs.detach().numpy()
        anomaly_scores = weighted_output.sum(axis=1)
        y_preds_np = np.ones(len(anomaly_scores))
        y_preds_np[anomaly_scores < 0] = -1

        y_preds_df = df.loc[y_preds_np == 1, ['timestamp', column]]
    
    anomalies_df = df.loc[df['anomaly'] == 1, ['timestamp', column]]
    
    ax.plot(df['timestamp'], df[column], color='blue', zorder=0)
    ax.scatter(anomalies_df['timestamp'], anomalies_df[column], color='red', edgecolor='k', marker='x', zorder=3)
    
    if isinstance(x, (np.ndarray, torch.Tensor)) and ensemble and meta_model:
        ax.scatter(y_preds_df['timestamp'] ,y_preds_df[column], color='yellow', edgecolor='k', s=50, zorder=2)
    
    if isinstance(targets, dict): 
        query_idxs = np.array(list(targets.keys()))
        ax.scatter(df['timestamp'].iloc[query_idxs], 
                   df[column].iloc[query_idxs], color='green', s=120, zorder=1)
    
    ax.set_title('Series')
    
    if isinstance(x, (np.ndarray, torch.Tensor)) and isinstance(targets, dict) and ensemble and meta_model:
        ax.legend(['Series','Predictions','True Anomalies','Queries'])
    elif isinstance(x, (np.ndarray, torch.Tensor)) and ensemble and meta_model:
        ax.legend(['Series','Predictions','True Anomalies'])
    else:
        ax.legend(['Series','True Anomalies'])
    
    if 'window' in df.columns:
        filler_idxs = df['window'].loc[df['window'].diff() != 0].index
        for i in range(1,len(filler_idxs),2):
            ax.axvspan(df['timestamp'].loc[filler_idxs[i]], df['timestamp'].loc[filler_idxs[i+1]], alpha=0.15, color='red')
    plt.show()
    
    
def plot_anomaly_series_grid(df, x=None, ensemble=None, meta_model=None, targets=None, window_size=10, column='lag0', boundary=0.1):
    """
    Plots the results of the series in df
    
    Args:
        df: pd.data_frame with columns 'timestamp', 'lag0', 'anomaly', ('window')
        x: numpy.array or torch.tensor of instances
        ensemble: ensemble model
        meta_model : meta model
        targets : dict containing previous targets
        window_size : number of lookback instances used (same as x.shape[1]?)
        column : column in containing actual series
        boundary : how much above and below the max/min should be colored
    """
    
    fig, ax = plt.subplots(figsize=(16,4))
    
    if isinstance(x, (np.ndarray, torch.Tensor)) and ensemble and meta_model:
        #Get the total model output
        if isinstance(x, torch.Tensor):
            ens_scores = ensemble.get_anomaly_score(x.numpy())
            mm_outputs = meta_model(x)
        elif isinstance(x, np.ndarray):
            ens_scores = ensemble.get_anomaly_score(x)
            mm_outputs = meta_model(torch.Tensor(x))
        else:
            raise ValueError('x should be either torch.Tensor or numpy.ndarray')

        weighted_output = ens_scores * mm_outputs.detach().numpy()
        anomaly_scores = weighted_output.sum(axis=1)
        y_preds_np = np.ones(len(anomaly_scores))
        y_preds_np[anomaly_scores < 0] = -1

        y_preds_df = df.loc[y_preds_np == 1, ['timestamp', column]]
    
    anomalies_df = df.loc[df['anomaly'] == 1, ['timestamp', column]]
    
    yaxis = [np.min(df[column]) - boundary, np.max(df[column]) + boundary]    
    yy = np.linspace(yaxis[0], yaxis[1], 50)
    
    grid_df = df.append([df]*(50-1)).sort_index()
    grid_df[column] = np.tile(yy, df.shape[0])
    
    train_cols = df.columns[-(window_size+1):]

    grid = grid_df[train_cols].to_numpy()
    
    ens_scores_grid = ensemble.get_anomaly_score(grid)
    mm_outputs_grid = (meta_model(torch.Tensor(grid))).detach().numpy()
    Z = (ens_scores_grid * mm_outputs_grid).sum(axis=1)
    Z = np.transpose(np.reshape(Z, (df.shape[0], len(yy))))
      
    ax.plot(df['timestamp'], df[column], color='blue', zorder=1)
    ax.scatter(anomalies_df['timestamp'], anomalies_df[column], color='red', edgecolor='k', marker='x', zorder=4)
    
    if isinstance(targets, dict): 
        query_idxs = np.array(list(targets.keys()))
        ax.scatter(df['timestamp'].iloc[query_idxs], 
                   df[column].iloc[query_idxs], color='green', s=120, zorder=2)
    
    if isinstance(x, (np.ndarray, torch.Tensor)) and ensemble and meta_model:
        ax.scatter(y_preds_df['timestamp'] ,y_preds_df[column], color='yellow', edgecolor='k', s=75, zorder=3)
    
    ax.set_title('Series')
    
    if isinstance(x, (np.ndarray, torch.Tensor)) and isinstance(targets, dict) and ensemble and meta_model:
        ax.legend(['Series','True Anomalies','Queries','Predictions'])
    elif isinstance(x, (np.ndarray, torch.Tensor)) and ensemble and meta_model:
        ax.legend(['Series','Predictions','True Anomalies'])
    else:
        ax.legend(['Series','True Anomalies'])
    
    if 'window' in df.columns:
        filler_idxs = df['window'].loc[df['window'].diff() != 0].index
        for i in range(1,len(filler_idxs),2):
            ax.axvspan(df['timestamp'].loc[filler_idxs[i]], df['timestamp'].loc[filler_idxs[i+1]], alpha=0.15, color='red')    
        
    ax.contourf(df['timestamp'], yy, Z, cmap=plt.cm.Blues, zorder=0)
    plt.show()

def plot_training(df, x=None, idxs=None, y_preds=None, anomaly_scores_evaluation=None ,targets=None, column='lag0', update=True, hlines=(0.5,0.25)):
    """
    Plots the training process
    
    Args:
        df: pd.data_frame with columns 'timestamp', 'lag0', 'anomaly', ('window')
        x: numpy.array or torch.tensor of instances
        idxs : indices we are currently training in x and df
        ensemble: ensemble model
        meta_model : meta model
        targets : dict containing previous targets
        column : column in containing actual series
        update : true  - dynamicly plot in notebook
                 false - does not dynamicly plot in notebook
        hlines : tuple (threshold for anoamlay, threshold for querry)
    """
    plt.figure(figsize=(16,9))
    #fig, ax = plt.subplots(figsize=(16,6))
    ax = plt.subplot2grid((16,6), (0,0), rowspan=6, colspan=16)
    ax2 = plt.subplot2grid((16,3), (7,0), rowspan=3, colspan=16, sharex=ax)
    
    #plt.ylim(-6., 3.)
    df_window = df.iloc[idxs[0]:idxs[1]]

    #y_preds_np = np.where(anomaly_scores > th, 1, -1)
    y_preds_df = df.loc[y_preds==1, ['timestamp', column]]
    anomalies_df = df.loc[df['anomaly'] == 1, ['timestamp', column]]
    
    ax.plot(df['timestamp'], df[column], color='blue', zorder=0)
    ax.plot(df_window['timestamp'], df_window[column], color='red', linewidth=2, zorder=1)
    ax.plot(df_window['timestamp'][-x.shape[1]:], df_window[column][-x.shape[1]:], color='green', linewidth=2, zorder=2)
    ax.scatter(anomalies_df['timestamp'], anomalies_df[column], color='red', edgecolor='k', marker='x', zorder=5)
    ax.scatter(y_preds_df['timestamp'], y_preds_df[column], color='yellow', edgecolor='k', s=50, zorder=4)
    
    if isinstance(targets, dict): 
        query_idxs = np.array(list(targets.keys()))
        ax.scatter(df['timestamp'].iloc[query_idxs],
                   df[column].iloc[query_idxs], color='green', edgecolor='k', s=120, zorder=3)
    
    ax.set_title('Training')
    
    ax2.plot(df['timestamp'].iloc[:idxs[1]], anomaly_scores_evaluation, color='purple')
    ax2.axhline(y=hlines[0], color='r', linestyle='-')
    ax2.axhline(y=hlines[1], color='g', linestyle='-')
    ax2.set_ylim(0, 4.)
    
    ax.legend(['Training Series', 'Current training', 'Sliding Window', 'Ture anomaly', 'Predicted anomaly', 'Queries'])
    if update:
        display.clear_output(wait=True)
        display.display(plt.gcf()) 
        time.sleep(0.01)
        plt.savefig('images/train{}.png'.format(1000 + idxs[0]))
    else:
        plt.show()
        
        
def plot_predictions(df, x=None, idxs=None, y_preds=None, anomaly_scores_evaluation=None ,targets=None, column='lag0', update=True, hlines=(0.5,0.25)):
    """
    Plots the training process
    
    Args:
        df: pd.data_frame with columns 'timestamp', 'lag0', 'anomaly', ('window')
        x: numpy.array or torch.tensor of instances
        idxs : indices we are currently training in x and df
        ensemble: ensemble model
        meta_model : meta model
        targets : dict containing previous targets
        column : column in containing actual series
        update : true  - dynamicly plot in notebook
                 false - does not dynamicly plot in notebook
        hlines : tuple (threshold for anoamlay, threshold for querry)
    """
    plt.figure(figsize=(16,9))
    #fig, ax = plt.subplots(figsize=(16,6))
    ax = plt.subplot2grid((16,6), (0,0), rowspan=6, colspan=16)
    ax2 = plt.subplot2grid((16,3), (7,0), rowspan=3, colspan=16, sharex=ax)
    
    #plt.ylim(-6., 3.)
    if idxs[0] < 0:
        df_window = df.iloc[:idxs[1]]
    else:
        df_window = df.iloc[idxs[0]:idxs[1]]

    #y_preds_np = np.where(anomaly_scores > th, 1, -1)
    y_preds_df = df.loc[y_preds==1, ['timestamp', column]]
    anomalies_df = df.loc[df['anomaly'] == 1, ['timestamp', column]]
    
    ax.plot(df['timestamp'], df[column], color='blue', zorder=0)
    ax.plot(df_window['timestamp'], df_window[column], color='red', linewidth=2, zorder=1)
    ax.plot(df_window['timestamp'][-x.shape[1]:], df_window[column][-x.shape[1]:], color='green', linewidth=2, zorder=2)
    ax.scatter(anomalies_df['timestamp'], anomalies_df[column], color='red', edgecolor='k', marker='x', zorder=5)
    ax.scatter(y_preds_df['timestamp'], y_preds_df[column], color='yellow', edgecolor='k', s=50, zorder=4)
    
    if isinstance(targets, dict): 
        query_idxs = np.array(list(targets.keys()))
        ax.scatter(df['timestamp'].iloc[query_idxs],
                   df[column].iloc[query_idxs], color='green', edgecolor='k', s=120, zorder=3)
    
    ax.set_title('Predicting')
    
    ax2.plot(df['timestamp'].iloc[:idxs[1]], anomaly_scores_evaluation, color='purple')
    ax2.axhline(y=hlines[0], color='r', linestyle='-')
    ax2.axhline(y=hlines[1], color='g', linestyle='-')
    ax2.set_ylim(0, 4.)
    
    ax.legend(['Training Series', 'Current training', 'Sliding Window', 'Ture anomaly', 'Predicted anomaly', 'Queries'])
    if update:
        display.clear_output(wait=True)
        display.display(plt.gcf()) 
        time.sleep(0.01)
        plt.savefig('images/test{}.png'.format(1000 + idxs[0]))
    else:
        plt.show()