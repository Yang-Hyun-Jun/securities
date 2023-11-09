
import torch

def get_r(result):
    sharpe = result["sharpe"]
    expect = result["expect"]
    sigma = result["sigma"]
    mdd = result["mdd"]
    
    # Calculate reward using sharpe, expect, sigma, mdd
    reward = sharpe * sigma - expect * mdd
    
    # Convert reward to tensor
    reward = torch.tensor([reward])
    
    return reward
