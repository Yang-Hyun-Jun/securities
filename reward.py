
import torch

def get_r(result):
    sharpe = result.get("sharpe", 0)
    expect = result.get("expect", 0)
    sigma = result.get("sigma", 0)
    mdd = result.get("mdd", 0)

    if sharpe > 0 and expect > 0 and sigma > 0 and mdd > 0:
        reward = sharpe * expect / (sigma * mdd)
    elif sharpe > 0 and expect > 0:
        reward = sharpe * expect
    elif sigma > 0 and mdd > 0:
        reward = sigma / mdd
    else:
        reward = 0

    reward = torch.tensor([reward])
    return reward
