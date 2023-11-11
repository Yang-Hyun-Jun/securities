import pandas as pd
import numpy as np
from factormanager import FactorManager

class Metrics:
    
    def get_mdd(self, pvs:list):
        """
        최대 낙폭
        """
        df = pd.DataFrame(pvs)
        premaxs = df.cummax()
        drawdowns = (1-df / premaxs) * 100
        mdd = drawdowns.max().iloc[0]
        return mdd
    
    def get_er(self, pvs:list):
        """
        기대 수익률
        """
        pvs = np.array(pvs)
        pct = (pvs[1:] - pvs[:-1]) / pvs[:-1]
        return np.mean(pct)

    def get_si(self, pvs:list):
        """
        변동성 
        """
        pvs = np.array(pvs)
        pct = (pvs[1:] - pvs[:-1]) / pvs[:-1]
        return np.std(pct)
    
    def get_sr(self, pvs:list):
        """
        샤프지수
        """
        free = (0.035) / 250
        pvs = np.array(pvs)
        pct = (pvs[1:] - pvs[:-1]) / pvs[:-1]
        ratio = np.mean(pct - free) / np.std(pct)
        return ratio