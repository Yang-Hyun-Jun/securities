import pandas as pd
import numpy as np
from numpy.random import random

class FactorManager:

    """
    투자 유니버스가 되는 종목들의 데이터를 관리한다.

    [Factors]
    'D3', 'D7', 'D14', 'NEWS'
    """

    path1 = 'data/factors.csv'
    path2 = 'data/price.csv'

    def __init__(self, config):    
        """
        all: 전 종목의 모든 팩터의 값을 담고 있는 데이터프레임
        price: 전 종목의 종가 값을 담고 있는 데이터프레임
        
        [Ex: all]
        종목코드  기준일       팩터     값
        AAPL  2023-01-03    D3      0.04707
        AAPL  2023-01-04    D7      0.0786
        ...     ...         ...     ...
        NVDA  2023-08-30    NEWS    0.008364
        """    
        self.all = pd.read_csv(self.path1, index_col=0, dtype={'종목코드':str})
        self.price = pd.read_csv(self.path2, index_col=0)
        self.factors = config['Factors']
        self.scores_by_date = []
    
    def get_FactorData(self, name:str) -> pd.DataFrame:
        """
        하나의 팩터 이름을 받아, 
        해당 팩터의 전 종목에 대한 값 데이터를 리턴

        
        [Ex: self.get_FactorData('D7')]
        종목코드      AAPL     NVDA     ACHC     ADTN  ... 
        기준일
        
        2023-01-03  0.03984 0.06998  0.03678  0.03482 ...   
        2023-01-04  0.03820 0.09536  0.09001  0.09153 ...
        ...
        2023-08-30
        """

        factor_data = self.all[self.all['팩터'] == name][['종목코드', '날짜', '값']]
        factor_data = factor_data.pivot(index='날짜', columns='종목코드', values='값')
        return factor_data
    
    def get_ScoreEACH(self, date:str) -> pd.DataFrame:
        """
        특정 Date에서 각 종목들의 팩터별 스코어 데이터 리턴

        
        [Ex: self.get_ScoreEACH('2023-01-31')]
        팩터       D3      D7   ...          NEWS
        종목코드
        AAPL      1.0     9.0   41.0   ...   45.0
        NVDA      13.0    5.0   13.0   ...   1.0 
        ...
        ACHC      8.0     16.0  20.0   ...   13.0
        """
        values = self.all[self.all['날짜'] == date][['종목코드', '팩터', '값']]
        values = values.pivot(index='종목코드', columns='팩터', values='값')

        factor_score = values[self.factors]
        factor_score = factor_score.apply(self.rankin_func)
        factor_score = factor_score.apply(self.minmax_func)
        factor_score = factor_score.apply(self.weight_func)
        return factor_score
    
    def get_RankALL(self):
        """
        팩터 점수를 합하여 토탈 랭킹 데이터 리턴

        
        [Ex: self.get_RankALL()]
        종목코드        AAPL  NVDA  ACHC   ADTN  ... 
        기준일
        
        2023-01-03    9.0     7.0     17.0    2.0 ...   
        2023-01-04    9.0     6.0     1.0     51.0 ...
        ...
        2023-08-30 
        """
        dates = self.price.index

        self.scores_by_date = [self.get_ScoreEACH(date)[self.factors] for date in dates] \
            if not self.scores_by_date else self.scores_by_date
        
        func1 = lambda df:df.apply(self.weight_func)
        func2 = lambda df:df.sum(axis=1).rank(method='first', ascending=False)

        rank_all = map(func1, self.scores_by_date)
        rank_all = map(func2, rank_all)
        rank_all = pd.concat(rank_all, axis=1).transpose()
        rank_all.index = dates
        return rank_all
    
    def set_w(self, value=None):
        """
        각 팩터의 가중치를 결정하는 함수
        """
        self.weight_dict = \
            dict(zip(self.factors, np.ones(len(self.factors)))) if value is None else \
            dict(zip(self.factors, value))
        

    def rankin_func(self, series):
        return series.rank(method='first')
    
    def weight_func(self, series):
        return series*self.weight_dict[series.name]
    
    def minmax_func(self, series):
        return (series-min(series)) / (max(series)-min(series)) + 1