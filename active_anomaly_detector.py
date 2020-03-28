from utils_plotting import *
from utils import my_loss

import torch
import numpy as np
import pandas as pd

from torch.utils import data
from sklearn import metrics

class AAD():
    def __init__(self, ensemble, meta_model, optimizer, scheduler=None, threshold=0.5, query_frequency=0.5, targets_to_keep=10):
        self.ensemble = ensemble
        self.meta_model = meta_model
        
        self.targets = dict()
        
        self.score_x = torch.zeros(len(self.targets), dtype=float)
        self.score_x_prev =torch.zeros(len(self.targets), dtype=float)
        
        #For starting prediciton function
        self.last_X_window = None
        self.last_y_window = None

        self.window_size = None
        self.mean_impute_rate = None
        
        self.threshold = threshold
        self.query_frequency = query_frequency
        
        self.targets_to_keep = targets_to_keep
        self.ensemble_update_counter = 0
        
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(self, X, y, y_anomalies, df, window_size=100, mean_impute_rate=0., update_prop=0.2, plot=True, print_metrics=True, return_metrics=False):
        """
        Fits the AAD with streaming data
        
        Args:
            X: numpy.array data, size n x MAX_WINDOW_SIZE
            y: numpy.array targets, size n
            y_anomalies : numpy.array of true anomalies {-1, 1}
            df: pd.data_frame with columns 'timestamp', 'lag0', 'anomaly' for plotting
            window_size : Number of instances to use when updating ensemble
            mean_impute_rate : Proportion of instances that should be imputed
            update_prop : Proportion of ensemble members to update at each ensemble update
            plot : True - plot progress in jupyter notebook
                   False - don't plot progress
            return_metrics : True - Prints metrics with finished
                             False - Does not print any metrics 
            return_metrics : True - Returns true anomalies and corresponding anomaly scores
                             False - Does not return any metrics
        """
        self.window_size = window_size
        self.mean_impute_rate = mean_impute_rate
        
        num_steps = len(X) - window_size
        
        X_window = X[:window_size]
        y_window = y[:window_size]

        #Fit ensemble to first window of data
        self.ensemble.fit(X=X_window, y=y_window, mean_impute_rate=mean_impute_rate)
        
        #For model evaluation and plotting
        anomaly_scores_evaluation = self.get_anomaly_scores(X_window, y_window)[:-1]
        predicted_anomalies = np.ones(len(y)) * -1
        
        for i in range(num_steps + 1):
            start_window, end_window = i, i + window_size
            
            #Current training window
            X_window = X[start_window: end_window]
            y_window = y[start_window: end_window]
            
            #Get the last and second last anomaly scores
            prev_anomaly_score = self.get_anomaly_scores(X_window[-2:-1], y_window[-2:-1])
            anomaly_score = self.get_anomaly_scores(X_window[-1:], y_window[-1:])
            
            #For model evaluation
            anomaly_scores_evaluation = np.concatenate((anomaly_scores_evaluation, anomaly_score))
            
            #Condition to querry
            if anomaly_score > self.threshold * self.query_frequency and anomaly_score/prev_anomaly_score > 1/(self.threshold * self.query_frequency):
                anomalous_idx = i + window_size - 1
                y_pred = 1 if anomaly_score > self.threshold else -1
                predicted_anomalies[anomalous_idx] = y_pred

                #Get the real label and store in targets
                y_real = y_anomalies[anomalous_idx]
                self.targets[anomalous_idx] = y_real
                
                #Remove old targets
                if len(self.targets) > self.targets_to_keep:
                    key_to_delete = min(self.targets)
                    del self.targets[key_to_delete]                
                
                #If wrong prediction -> update ensemble and meta model
                #If right prediction -> update meta model with correct_pred = True
                if y_real != y_pred:
                    self.update_ensemble(X_window, y_window, update_prop=update_prop, mean_impute_rate=mean_impute_rate)
                    self.update_aad(X[:end_window], y[:end_window])
                    self.ensemble_update_counter += 1
                else:
                    self.update_aad(X[:end_window], y[:end_window], correct_pred=True)
                    
            #Plots the training process
            if plot:
                plot_training(df=df,
                               x=X_window,
                               idxs=(start_window, end_window),
                               y_preds=predicted_anomalies,
                               anomaly_scores_evaluation=anomaly_scores_evaluation,
                               targets=self.targets,
                               update=True,
                               hlines=(self.threshold, self.threshold * self.query_frequency))
        
        #Store last window for prediction
        self.last_X_window = X[-window_size:]
        self.last_y_window = y[-window_size:]
        
        #Calculate and print metrics
        if print_metrics:
            auc = metrics.roc_auc_score(y_anomalies, anomaly_scores_evaluation)
            print("auc", auc)
            f1 = metrics.f1_score(y_anomalies, predicted_anomalies)
            print("f1 :", f1)
        
        if return_metrics:
            return (y_anomalies, anomaly_scores_evaluation, predicted_anomalies)
        
    def predict(self, X, y, y_anomalies, df, window_size=100, mean_impute_rate=0., update_prop=0.2, plot=True, print_metrics=True, return_metrics=False):
        """
        Predicts using the AAD with streaming data
        
        Args:
            X: numpy.array data, size n x MAX_WINDOW_SIZE
            y: numpy.array targets, size n
            y_anomalies : numpy.array of true anomalies {-1, 1}
            df: pd.data_frame with columns 'timestamp', 'lag0', 'anomaly' for plotting
            window_size : Number of instances to use when updating ensemble
            mean_impute_rate : Proportion of instances that should be imputed
            update_prop : Proportion of ensemble members to update at each ensemble update
            plot : True - plot progress in jupyter notebook
                   False - don't plot progress
            return_metrics : True - Prints metrics with finished
                             False - Does not print any metrics       
            return_metrics : True - Returns true anomalies and corresponding anomaly scores
                             False - Does not return any metrics
        """
        num_steps = len(X)
        
        X_window = self.last_X_window
        y_window = self.last_y_window
        
        #For model evaluation
        anomaly_scores_evaluation = np.array([])
        predicted_anomalies = np.ones(len(y)) * -1
        
        #Resetting parameters
        self.targets = dict()
        self.score_x = torch.zeros(len(self.targets), dtype=float)
        self.score_x_prev =torch.zeros(len(self.targets), dtype=float)
        
        
        for i in range(1, num_steps + 1):
            start_window, end_window = i - window_size, i
            
            #Current training window
            if start_window < 0:
                X_window = np.concatenate([self.last_X_window[end_window:], X[:end_window]])
                y_window = np.concatenate([self.last_y_window[end_window:], y[:end_window]])
            else:
                X_window = X[start_window: end_window]
                y_window = y[start_window: end_window]
            
            #Get the last and second last anomaly scores
            prev_anomaly_score = self.get_anomaly_scores(X_window[-2:-1], y_window[-2:-1])
            anomaly_score = self.get_anomaly_scores(X_window[-1:], y_window[-1:])
            
            #For model evaluation
            anomaly_scores_evaluation = np.concatenate((anomaly_scores_evaluation, anomaly_score))

            #Condition to querry
            if anomaly_score > self.threshold*self.query_frequency and anomaly_score/prev_anomaly_score > 1/(self.threshold*self.query_frequency):
                anomalous_idx = i - 1
                y_pred = 1 if anomaly_score > self.threshold else -1
                predicted_anomalies[anomalous_idx] = y_pred
                
                #Get the real label and store in targets
                y_real = y_anomalies[anomalous_idx]
                self.targets[anomalous_idx] = y_real

                #Remove old targets
                if len(self.targets) > self.targets_to_keep:
                    key_to_delete = min(self.targets)
                    del self.targets[key_to_delete]
                
                #If wrong prediction -> update ensemble and meta model
                #If right prediction -> update meta model with correct_pred = True
                if y_real != y_pred:
                    self.update_ensemble(X_window, y_window, update_prop=update_prop, mean_impute_rate=mean_impute_rate)
                    self.update_aad(X[:end_window], y[:end_window])                    
                    self.ensemble_update_counter += 1
                else:
                    self.update_aad(X[:end_window], y[:end_window], correct_pred=True)
                    
            #Plots the training process
            if plot:
                plot_predictions(df=df,
                                 x=X_window,
                                 idxs=(start_window, end_window),
                                 y_preds=predicted_anomalies,
                                 anomaly_scores_evaluation = anomaly_scores_evaluation,
                                 targets=self.targets,
                                 update=True,
                                 hlines=(self.threshold, self.threshold * self.query_frequency))
            
        if print_metrics:
            auc = metrics.roc_auc_score(y_anomalies, anomaly_scores_evaluation)
            print("auc", auc)
            f1 = metrics.f1_score(y_anomalies, predicted_anomalies)
            print("f1 :", f1)   
        if return_metrics:
            return (y_anomalies, anomaly_scores_evaluation, predicted_anomalies)
        
    def get_predictions(self, X, y):
        """
        Gets anomaly predictions for the aad {-1,1}
        
        Args:
            X: numpy.array data, size n x MAX_WINDOW_SIZE
            y: numpy.array targets, size n
        
        Returns:
            predictions: numpy.array containing {-1 (normal), 1(anomaly)}
        """
        X_tensor = torch.Tensor(X)
        
        ens_scores = self.ensemble.get_anomaly_score(X, y)
        mm_outputs = self.meta_model(X_tensor)

        X_scores = (ens_scores * mm_outputs.detach().numpy()).sum(axis=1)
        predictions = np.where(X_scores > self.threshold, 1, -1)

        return predictions 
    
    def get_anomaly_scores(self, X, y):
        """
        Gets anomaly scores for the aad
        
        Args:
            X: numpy.array data, size n x MAX_WINDOW_SIZE
            y: numpy.array targets, size n
        
        Returns:
            X_scores: numpy.array containing anomaly scores [0, inf]
        """
        X_tensor = torch.Tensor(X)
        
        ens_scores = self.ensemble.get_anomaly_score(X, y)
        mm_outputs = self.meta_model(X_tensor)

        X_scores = (ens_scores * mm_outputs.detach().numpy()).sum(axis=1)

        return X_scores
   
    
    def update_ensemble(self, X, y, mean_impute_rate, update_prop=0.2):
        """
        Retrains the oldest models in the ensemble
        
        Args:
            X: numpy.array data, size n x MAX_WINDOW_SIZE
            y: numpy.array targets, size n
            mean_impute_rate : Proportion of instances that should be imputed
            update_prop : Proportion of ensemble members to update at each ensemble update
        """
        num_models = len(self.ensemble.LinearRegression_models)
        
        #Number of partions in the data
        num_partitions = int(1/update_prop)
        #Last partition to be changed
        i = self.ensemble_update_counter % num_partitions
        #Divides the indexs in to ´num_partitions´ partitions
        idxs = np.split(np.arange(num_models), num_partitions)
        
        self.ensemble.update_ensemble_models(X, y, idxs[i], mean_impute_rate)
    
    def update_aad(self, X, y, correct_pred=False):
        """
        Trains the meta model
        
        Args:
            X: numpy.array data, size n x MAX_WINDOW_SIZE
            y: numpy.array targets, size n
            correct_pred: True - run 10 epochs even though all querried labels are corectly classified
                          False - run training until all querried labels are corectly classified
        """
        #Get the labels and indexes for the target
        queried_labels = np.array(list(self.targets.values()))
        idxs_of_labels = torch.tensor(list(self.targets.keys()))     
        
        #Get the input and target for querried instances
        x_targets = X[idxs_of_labels].reshape((-1, X.shape[1]))
        y_targets = y[idxs_of_labels].reshape((-1,))
        
        #To torch tensor
        tensor_x_targets = torch.Tensor(x_targets) 
        tensor_y_targets = torch.Tensor(y_targets)
        
        #Set Batch Size to same size as querried instances (max 10)
        batch_size = len(self.targets)
        
        train_ds_targets = data.TensorDataset(tensor_x_targets, tensor_y_targets) # create datset
        train_dl_targets = data.DataLoader(train_ds_targets, batch_size=batch_size, shuffle=False) # create dataloader
        
        #Get perdictions for the labeled instances 
        labeled_preds = self.get_predictions(x_targets, y_targets)
        
        #Get previous anomaly scores for loss function, adds zeros if less than 10 querries has been made
        self.score_x_prev = torch.cat([self.score_x, torch.zeros(len(self.targets)-len(self.score_x), dtype=float)])
        
        epoch = 0
        while not all(labeled_preds == queried_labels):

            #Get perdictions for the labeled instances        
            labeled_preds = self.get_predictions(x_targets, y_targets)
            epoch_loss = 0

            for i, [inputs, targets] in enumerate(train_dl_targets):
                #Save ensemble input as numpy
                ens_input = inputs.numpy()
                ens_target = targets.numpy()

                #Load with gradient accumulation capabilities
                inputs = inputs.requires_grad_()

                #Clear gradients w.r.t. parameters
                self.optimizer.zero_grad()

                #Forward pass to get output (logits)
                mm_outputs = self.meta_model(inputs)

                #Get corresponging anomaly scores
                ens_scores = self.ensemble.get_anomaly_score(ens_input, ens_target)
                
                #Calculate the anomaly scores
                self.score_x = (torch.tensor(ens_scores) * mm_outputs).sum(dim=1)
                
                # Calculate Loss
                loss = my_loss(mm_outputs, self.targets, self.score_x, self.score_x_prev, q_tau=self.threshold)
                epoch_loss += loss.item()
                                
                #Store old score x
                self.score_x_prev = self.score_x
                
                #Getting gradients w.r.t. parameters
                loss.backward(retain_graph=True)

                #Updating parameters
                self.optimizer.step()
            
            #Update learning rate
            self.scheduler.step()
            
            epoch += 1
            # Print Loss every 10th epoch
            if epoch % 10 == 0:
                print('Epoch: {}, Loss: {}'.format(epoch, epoch_loss/len(train_dl_targets)))
        
        #Run 10 epochs even though all querried labels are corectly classified to cement predictions
        if correct_pred:
            for epoch in range(10):
                epoch_loss = 0
                for i, [inputs, targets] in enumerate(train_dl_targets):
                    # Save ensemble input as numpy
                    ens_input = inputs.numpy()
                    ens_target = targets.numpy()

                    # Load with gradient accumulation capabilities
                    inputs = inputs.requires_grad_()

                    # Clear gradients w.r.t. parameters
                    self.optimizer.zero_grad()

                    # Forward pass to get output/logits
                    mm_outputs = self.meta_model(inputs)

                    # Get corresponging anomaly scores
                    ens_scores = self.ensemble.get_anomaly_score(ens_input, ens_target)

                    weighted_output = (torch.tensor(ens_scores) * mm_outputs)
                    self.score_x = weighted_output.sum(dim=1)

                    # Calculate Loss
                    loss = my_loss(mm_outputs, self.targets, self.score_x, self.score_x_prev, q_tau=self.threshold)
                    epoch_loss += loss.item()

                    #Store score x
                    self.score_x_prev = self.score_x

                    # Getting gradients w.r.t. parameters
                    loss.backward(retain_graph=True)

                    # Updating parameters
                    self.optimizer.step()
                
                #Update learning rate
                self.scheduler.step()