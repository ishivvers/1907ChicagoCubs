
import numpy as np

def evaluate( model, data ):
    """
    A function to get mean absolute error in order to evaluate the model
    s and e are the number of sites and events (days)
    """
    
    s = len(data)
    try:
        e = len(data[0])
    except:
        e = 1.0
    abs_dif = abs(data - model)

    return sum( abs_dif)/(s*e)
