from init_objects import *
from qe_model import *
import time

start_time = time.time()

# # 1 setup parameters
parameters = {"fundamental_value": 105,
              "trader_sample_size": 15, "n_traders": 500,
              "ticks": 500, "std_fundamental": 0.01,
              "std_noise": 0.159, "w_random": 0.25,
              "strat_share_chartists": 0.20,
              "init_assets": 740, "base_risk_aversion": 4.051,
              'spread_max': 0.004, "horizon": 200,
              "fundamentalist_horizon_multiplier": 2.2,
              "trades_per_tick": 3, "mutation_intensity": 0.0477,
              "average_learning_ability": 0.02, 'money_multiplier': 2.6,
              "bond_mean_reversion": 0.0, 'cb_pf_range': 0.05,
              "qe_perc_size": 0.16, "cb_size": 0.02, "qe_asset_index": 0}


seed = 0
# 2 initialise model objects
traders, central_bank, orderbook = init_objects(parameters, seed)

# 3 simulate model
traders, central_bank, orderbook = qe_model(traders, central_bank, orderbook, parameters, scenario='BLR', seed=seed)

print("The simulations took", time.time() - start_time, "to run")



