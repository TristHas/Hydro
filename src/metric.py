import numpy as np


def nse(output,observed):
    output = output.reshape(-1,)
    observed = observed.reshape(-1,)
    return 1 - mse(output,observed)/np.var(observed)


def kge(output,observed):
    output = output.reshape(-1,)
    observed = observed.reshape(-1,)
    r = np.corrcoef(output,observed)[0,1]
    a = np.std(output)/np.std(observed)
    b = np.mean(output)/np.mean(observed)
    return 1 - np.sqrt((r-1)**2 + (a-1)**2 + (b-1)**2)


def mse(output,observed):
    output = output.reshape(-1,)
    observed = observed.reshape(-1,)
    return np.mean((np.square(output-observed)))

