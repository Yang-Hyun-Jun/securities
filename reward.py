import torch

def get_r(result):
    sharpe = result['sharpe']
    expect = result['expect']
    sigma = result['sigma']
    mdd = result['mdd']
    
    reward = sharpe + (expect / sigma) - (mdd * sigma)
    reward = torch.tensor([reward])
    
    return reward
