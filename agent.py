import torch
import random

from collections import deque
from backtester import BackTester
from torch.optim import Adam
from torch.nn import MSELoss
from network import Mask
from network import Rnet
from reward import get_r

torch.set_printoptions(sci_mode=False)

device = 'cpu'

class RLSEARCH(BackTester):
    def __init__(self, config):
        BackTester.__init__(self, config)

        dim = config['Dim']
        self.mnet = Mask(dim).to(device)
        self.rnet = Rnet(dim).to(device)
        self.mse = MSELoss()

        self.opt_r = Adam(self.rnet.parameters(), lr=1e-4)
        self.opt_a = Adam(self.mnet.parameters(), lr=8e-3)

        self.w_tensor = deque(maxlen=500)
        self.r_tensor = deque(maxlen=500)

    def save(self, path):
        torch.save(self.mnet.state_dict(), path)
        torch.save(self.rnet.state_dict(), path)

    def get_w(self, noise=True):
        """
        Policy로부터 팩터 가중치 샘플링
        """
        return self.mnet.sample(noise).cpu()

    def get_r(self, result:dict):
        """
        결과 메트릭으로부터 reward 계산
        """
        reward = result['profit']
        reward = torch.tensor([reward])
        return reward
        
    def update(self, w, r):
        """
        리버스 엔지니어링 업데이트
        """
        # R network update
        r_hat = self.rnet(w.detach())
        r_loss = self.mse(r_hat, r)
        
        self.opt_r.zero_grad()
        r_loss.backward()
        self.opt_r.step()

        # Policy update
        w_loss = -self.rnet(w).mean()
        self.opt_a.zero_grad()
        w_loss.backward(retain_graph=True)
        self.opt_a.step()
        return r_loss.item(), w_loss.item()
        
    def search(self, iter, start='2023-01', end='2023-08'):
        """
        RL 에이전트 학습 Loop
        """

        batch_size = 32

        for i in range(iter):
            weight = self.get_w()
            self.init(weight.detach().numpy())
            result = self.test(start, end)[-1]
            reward = self.get_r(result)

            self.w_tensor.append(weight)
            self.r_tensor.append(reward)

            if len(self.w_tensor) >= batch_size:
                w_batch = random.sample(self.w_tensor, batch_size)
                r_batch = random.sample(self.r_tensor, batch_size)
                w_batch = torch.stack(w_batch).float().to(device)
                r_batch = torch.stack(r_batch).float().to(device)
                self.update(w_batch, r_batch)