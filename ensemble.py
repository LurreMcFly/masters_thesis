import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

from scipy.ndimage.interpolation import shift

import warnings
warnings.filterwarnings('ignore')

class Ensemble():
    
    def __init__(self, ensemble_dict):
        self.ensemble_dict = ensemble_dict
        self.ensemble_models = []
        
        self.read_dict()
        
        if initiate:
            self.initiate_enseble_members()
        
    def __str__(self):
        string = ['Ensamble consists of :']
        for ensemble_model in self.ensemble_models:
            string.append('\n    {}'.format(ensemble_model))
            
        return '\n'.join(string)    
    
    def __len__(self):
        return len(self.ensemble_models)    
    
    
    def read_dict(self):
        """
        reads the ensemble_dict for what models to use
        in the ensemble
        """
        self.__dict__
        
        for k,v in self.ensemble_dict.items():
            
            #check if invalid ensemble_dict
            if k not in self.__dict__:
                raise ValueError('{} is not a valid ensemble member'.format(k))
            
            #check if valud values for key
            if not isinstance(v, list):
                raise ValueError('Value {} is not a list for ensemble member {} in ensemble_dict'.format(v, k))
                
            
            self.__dict__[k] = v

class Ensemble_IsolationForest(Ensemble):
    
    def __init__(self, ensemble_dict, initiate=True):
        self.ensemble_dict = ensemble_dict
        self.IsolationForest = None
        self.IsolationForest_models = None
        
        self.ensemble_models = []
        
        self.read_dict()
        
        if initiate:
            self.initiate_enseble_members()
            
    def initiate_IsolationForest(self, **kwargs):
        """
        Initiates Isolation Forset members according to the list values for IsolationForest.
        Length of list specifies number of Isolation Forests
        Number at place i in list specifies number of trees within the IsolationForest
        ----------
        """
        self.IsolationForest_models = []

        for n_trees in self.IsolationForest:
            isolation_forest = IsolationForest(n_estimators=n_trees, 
                                               behaviour='new',
                                               **kwargs)
            self.IsolationForest_models.append(isolation_forest)
        
        self.ensemble_models += self.IsolationForest_models
        
        
    def initiate_enseble_members(self):
        """
        Initiates all models with default settings
        """
        self.initiate_IsolationForest()        
    
    def fit(self, X):
        """
        Fits all models in ensemble to data X.
        
        Assumes each ensemble member has a fit function.        
        """
        for ensemble_member in self.ensemble_models:
            ensemble_member.fit(X)
            
    def fit_bootstrap(self, X):
        """
        Fits all models in ensemble to data X.
        
        Assumes each ensemble member has a fit function.        
        """
        for ensemble_member in self.ensemble_models:
            idxs = np.random.choice(np.arange(len(X)), size=len(X))
            ensemble_member.fit(X[idxs])
    
    
    def predict(self, X):
        """
        Makes predictions for all models in ensemble.
        Assumes each ensemble member has a predict function.        
        """
        n = X.shape[0]
        
        y_pred = np.empty([n, len(self)], dtype=float)
        
        for i, ensemble_member in enumerate(self.ensemble_models):
            y_pred_member = ensemble_member.predict(X)
            y_pred[:,i] = y_pred_member
            
        return y_pred
    
    def predict_by_score(self, X):
        """
        Makes predictions for all models in ensemble.
        Makes predictions based on function get_anomaly_score.
        Should equal function predict
        """
        m = X.shape[0]
        
        y_score = self.get_anomaly_score(X)
        y_pred = np.empty([n, len(self)], dtype=float)
        
        for i, score in enumerate(y_score):
            y_pred_member = np.ones(m)
            y_pred_member[score < 0] = -1
            y_pred[:,i] = y_pred_member
        
        return y_preds
    
    def get_anomaly_score(self, X):
        """
        Gets anomaly score for all models in ensemble.
        
        Assumes each ensemble member has a function for the score.
        """
        n = X.shape[0]
        
        y_score = np.empty([n, len(self)], dtype=float)
        for i, ensemble_member in enumerate(self.ensemble_models):
            y_score_member = -ensemble_member.decision_function(X)
            y_score[:,i] = y_score_member
        
        return y_score
    
    def update_ensemble_models(self, X, idxs, **kwargs):
        """
        Updates the ensemble memebers at idxs
        """
        for i in idxs:
            ensemble_model = self.ensemble_models[int(i)]
            if isinstance(ensemble_model, IsolationForest):
                n_trees = self.IsolationForest[int(i)]
                self.ensemble_models[int(i)] = IsolationForest(n_estimators=n_trees, 
                                                               behaviour='new',
                                                               **kwargs)
                self.ensemble_models[int(i)].fit(X)
                
                
                
class Ensemble_LinearRegression(Ensemble):
    
    def __init__(self, ensemble_dict, initiate=True):
        self.ensemble_dict = ensemble_dict
        self.ensemble_models = []
        
        self.window_sizes = self.ensemble_dict['LinearRegression']
        self.LinearRegression_models = []
        
        self.max_window_size = max(self.window_sizes)
        self.last_max_window = None

        if initiate:
            self.initiate_LinearRegression()
    
    def initiate_LinearRegression(self, **kwargs):
        """
        Initiates Liear Regression models according to the sliding window lengths in self.window_sizes.
        Length of list specifies number of  Liear Regression models
        Number at place i in self.window_sizes specifies lengths of sliding window used for model i 
        """
        for _ in self.window_sizes:
            
            linear_regression_model = LinearRegression(**kwargs)
            self.LinearRegression_models.append(linear_regression_model)
        
        self.ensemble_models += self.LinearRegression_models
        
    def fit(self, X, y, mean_impute_rate=0):
        """
        Fits all models in ensemble to data X.
        """
        for window_size, model in zip(self.window_sizes, self.LinearRegression_models):
            
            if mean_impute_rate:
                impute_idxs = np.random.randint(1, len(X) - 1, size=int(len(X)*mean_impute_rate))
                X_imputed = X.copy()
                for impute_idx in impute_idxs:
                    np.fill_diagonal(X_imputed[impute_idx:], X_imputed[impute_idx - 1, 0])
                
                x_window = X_imputed[:, :window_size]
            else:
                x_window = X[:, :window_size]
            
            model.fit(x_window, y)
    
    def generate_sliding_window(self, X, window_size):
        """
        Gererates sliding windows of X and corresponding targets
        """
        x_out = np.empty([len(X), window_size])
        y = X[window_size:]
        
        for i in range(window_size):
            x_out[:, i] = shift(X, i)
            
        return x_out[window_size-1:-1], y
        
    def predict(self, X, start=0):
        """
        Makes predictions for all models in ensemble.
        Assumes each ensemble member has a predict function.
        """        
        predictions = np.empty([len(X), len(self.window_sizes)])
        
        for i, (window_size, model) in enumerate(zip(self.window_sizes, self.LinearRegression_models)):
            x_window = X[:, :window_size]
            predictions[:,i] = model.predict(x_window)
        
        return predictions
    
    def get_anomaly_score(self, X, y, start=0):
        """
        Gets anomaly score for all models in ensemble.       
        """              
        anomaly_scores = np.empty([len(X), len(self.window_sizes)])
        
        for i, (window_size, model) in enumerate(zip(self.window_sizes, self.LinearRegression_models)):
            x_window = X[:, :window_size]       
            predictions = model.predict(x_window)
            anomaly_scores[:,i] = np.abs(y - predictions)
        
        return anomaly_scores
    
    def update_ensemble_models(self, X, y, idxs, mean_impute_rate, **kwargs):
        """
        Updates the ensemble memebers at idxs
        """     
        for i in idxs:
            idx = int(i)
            ensemble_model = self.ensemble_models[idx]
            if isinstance(ensemble_model, LinearRegression):
                window_size = self.window_sizes[idx]
                
                if mean_impute_rate:
                    impute_idxs = np.random.randint(1, len(X) - 1, size=int(len(X)*mean_impute_rate))
                    X_imputed = X.copy()
                    for impute_idx in impute_idxs:
                        np.fill_diagonal(X_imputed[impute_idx:], X_imputed[impute_idx - 1, 0])
                    x_window = X_imputed[:,  -window_size:]
                else:
                    x_window = X[:, -window_size:]                
                
                self.ensemble_models[idx].fit(x_window, y)