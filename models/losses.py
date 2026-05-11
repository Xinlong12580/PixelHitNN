from tensorflow.math import log
from tensorflow import print as tfprint
from tensorflow import reduce_mean, maximum

def nll_loss(y_true,y_pred):
    y_position = y_pred[:,0]
    y_uncertainty = y_pred[:,1]
    y_target = y_true[:,0]
    y_variance = y_uncertainty ** 2

    y_log_var = log(y_variance)
    #Eq. 4 https://arxiv.org/pdf/2109.08213.pdf
    return reduce_mean(y_log_var+(y_target-y_position)**2/y_variance)

def nll_loss_debug(y_true,y_pred):
    #For debugging nan https://stackoverflow.com/questions/38810424/how-does-one-debug-nan-values-in-tensorflow
    y_position  = y_pred[:,0]
    y_variance  = y_pred[:,1]
    y_target    = y_true[:,0]
    y_log_var   = log(y_variance)

    tfprint(y_pred)
    tfprint(y_position)
    tfprint(y_target)
    tfprint(y_log_var)
    tfprint(y_variance)
    quit()

    #Eq. 4 https://arxiv.org/pdf/2109.08213.pdf
    return reduce_mean(y_log_var+(y_target-y_position)**2/y_variance)

def mse_position(y_true,y_pred):
    y_position  = y_pred[:,0]
    y_target    = y_true[:,0]

    return reduce_mean((y_target - y_position)**2)

def mean_pulls(y_true,y_pred):
    y_position  = y_pred[:,0]
    y_target    = y_true[:,0]
    y_variance  = y_pred[:,1]

    return reduce_mean((y_target-y_position)**2/y_variance)

