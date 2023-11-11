import argparse
import pandas as pd 

from agent import RLSEARCH

all = ['D3', 'D7', 'D14', 'NEWS']

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--number", type=int, default=30)
parser.add_argument("--balance", type=int, default=1000)
parser.add_argument("--algorithm", type=str, default='RL')
parser.add_argument("--quintile", type=int, default=1)
parser.add_argument("--quarter", type=str, default='1Q')
parser.add_argument("--factors", nargs='*', default=all)
args = parser.parse_args()


if __name__ == '__main__':

    config = {'Number': args.number, 
              'Quantile': args.quintile,
              'Balance': args.balance,
              'Quarter': args.quarter,
              'Factors': args.factors,
              'Dim': len(args.factors)}


    if args.algorithm == 'RL':
        RLsearch = RLSEARCH(config)
        RLsearch.search(100, '2023-01', '2023-06')

        optimal = RLsearch.get_w(False)
        optimal = optimal.detach().numpy()
        RLsearch.init(optimal)
        PVs, PFs, TIs, POs, result = RLsearch.test('2023-06', '2023-09')

    seed = args.seed
    algo = args.algorithm
    pd.DataFrame(PVs).to_csv(f'result/seed{seed}/PV_{algo}.csv')
    pd.DataFrame(PFs).to_csv(f'result/seed{seed}/PF_{algo}.csv')
    pd.DataFrame(TIs).to_csv(f'result/seed{seed}/TI_{algo}.csv')
    pd.DataFrame(POs).to_csv(f'result/seed{seed}/PO_{algo}.csv')
    pd.DataFrame(optimal.reshape(1,-1))\
        .to_csv(f'result/seed{seed}/We_{algo}.csv')
    pd.DataFrame.from_dict(result, orient='index')\
        .to_csv(f'result/seed{seed}/Me_{algo}.csv')

    
