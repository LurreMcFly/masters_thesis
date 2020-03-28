import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Suppression_Layer(nn.Linear):
    """
    Layer used in the FSSN from the paper
    Das, S. and Doppa, J.R. (2018). 
    GLAD: GLocalized Anomaly Detection via Active Feature Space Suppression. 
    (https://arxiv.org/abs/1810.01403)
    
    Linear layer using kaiming/xavier initialization
    
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    
    """
    def __init__(self, *args):
        super(Suppression_Layer, self).__init__(*args)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=math.sqrt(3))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
class FSSN_softmax(nn.Module):
    """
    Feature Space Suppression Network (FSSN)
    Das, S. and Doppa, J.R. (2018). 
    GLAD: GLocalized Anomaly Detection via Active Feature Space Suppression. 
    (https://arxiv.org/abs/1810.01403)
    
    With softmax output activation function
    
    "a shallow neural network with max(50, 3M) hidden nodes for all our test datasets, 
    where M is the number ofensemble members" pg 37 in paper
    
    """
    
    def __init__(self, n_inputs, n_ensemble, n_nodes=50):
        super(FSSN, self).__init__()
        
        self.sup1 = Suppression_Layer(n_inputs, max(n_nodes, 3*n_ensemble))
        self.out = Suppression_Layer(max(n_nodes, 3*n_ensemble), n_ensemble)
        
    def forward(self, x):
        x = torch.sigmoid(self.sup1(x))
        x = F.softmax(self.out(x), dim=1)
        
        return x
    
    
class FSSN(nn.Module):
    """
    Feature Space Suppression Network (FSSN)
    Das, S. and Doppa, J.R. (2018). 
    GLAD: GLocalized Anomaly Detection via Active Feature Space Suppression. 
    (https://arxiv.org/abs/1810.01403)
    
    With sigmoid output activation function
    
    "a shallow neural network with max(50, 3M) hidden nodes for all our test datasets, 
    where M is the number ofensemble members" pg 37 in paper
    
    """
    
    def __init__(self, n_inputs, n_ensemble, n_nodes=50):
        super(FSSN, self).__init__()
        
        self.sup1 = Suppression_Layer(n_inputs, max(n_nodes, 3*n_ensemble))
        self.out = Suppression_Layer(max(n_nodes, 3*n_ensemble), n_ensemble)
        
    def forward(self, x):
        x = torch.sigmoid(self.sup1(x))
        x = F.sigmoid(self.out(x))
        
        return x