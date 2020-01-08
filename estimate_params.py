from functions.indirect_calibration import *
from init_objects import *
from qe_model import *
import time
from multiprocessing import Pool
import json
import numpy as np
import math
from hurst import compute_Hc

np.seterr(all='ignore')

start_time = time.time()

# INPUT PARAMETERS
LATIN_NUMBER = 1
NRUNS = 3
BURN_IN = 0
CORES = NRUNS # set the amount of cores equal to the amount of runs

problem = {
  'num_vars': 7,
  'names': ['std_noise', "w_random", "strat_share_chartists",
            "base_risk_aversion", "fundamentalist_horizon_multiplier",
            "mutation_intensity", "average_learning_ability"],
  'bounds': [[0.03, 0.09], [0.02, 0.15], [0.02, 0.3],
            [0.05, 3.0], [1.0, 5.0],
            [0.05,0.5], [0.01, 0.8]]
}

with open('hypercube.txt', 'r') as f:
    latin_hyper_cube = json.loads(f.read())

# Bounds
LB = [x[0] for x in problem['bounds']]
UB = [x[1] for x in problem['bounds']]

init_parameters = latin_hyper_cube[LATIN_NUMBER]

params = {"fundamental_value": 166,
             "trader_sample_size": 22,
             "n_traders": 1000,
             "ticks": 600,
             "std_fundamental": 0.053,
             "init_assets": 740,
             'spread_max': 0.004,
             'money_multiplier': 2.2,
             "horizon": 200,
             "std_noise": 0.049,
             "w_random": 0.08,
             "strat_share_chartists": 0.08,
             "base_risk_aversion": 1.051,
             "fundamentalist_horizon_multiplier": 3.8,
             "trades_per_tick": 1, "mutation_intensity": 0.0477,
             "average_learning_ability": 0.05,
             "bond_mean_reversion": 0.0, 'cb_pf_range': 0.05,
             "qe_perc_size": 0.16, "cb_size": 0.02, "qe_asset_index": 0, "qe_start": 2, "qe_end":598}


def simulate_a_seed(seed_params):
    """Simulates the model for a single seed and outputs the associated cost"""
    seed = seed_params[0]
    params = seed_params[1]

    obs = []
    # run model with parameters
    traders, central_bank, orderbook = init_objects(params, seed)
    traders, central_bank, orderbook = qe_model(traders, central_bank, orderbook, params, scenario='None',
                                                seed=seed)
    obs.append(orderbook)

    # store simulated stylized facts
    mc_prices, mc_returns, mc_autocorr_returns, mc_autocorr_abs_returns, mc_volatility, mc_volume, mc_fundamentals = organise_data(
        obs)

    autocor = []
    autocor_abs = []
    kurtosis = []
    hursts = []

    for col in mc_returns:
        autocor.append(autocorrelation_returns(mc_returns[col][1:], 25).mean())
        autocor_abs.append(autocorrelation_abs_returns(mc_returns[col][1:], 25).mean())
        kurtosis.append(mc_returns[col][1:].kurtosis())
        H, c, data = compute_Hc(mc_prices[col].dropna(), kind='price', simplified=True)
        hursts.append(H)

    stylized_facts_sim = np.array([
        np.mean(autocor),
        np.mean(autocor_abs),
        np.mean(kurtosis),
        np.mean(hursts)
    ])

    W = np.load('distr_weighting_matrix.npy')  # if this doesn't work, use: np.identity(len(stylized_facts_sim))

    empirical_moments = np.array([0.05034916, 0.06925489, 4.16055312, 0.71581425])

    # calculate the cost
    cost = quadratic_loss_function(stylized_facts_sim, empirical_moments, W)
    return cost


def pool_handler():
    p = Pool(CORES) # argument is how many process happening in parallel
    list_of_seeds = [x for x in range(NRUNS)]

    def model_performance(input_parameters):
        """
        Simple function calibrate uncertain model parameters
        :param input_parameters: list of input parameters
        :return: average cost
        """
        # convert relevant parameters to integers
        new_input_params = []
        for idx, par in enumerate(input_parameters):
            new_input_params.append(par)

        # update params
        uncertain_parameters = dict(zip(problem['names'], new_input_params))
        params = {"fundamental_value": 166,
             "trader_sample_size": 22,
             "n_traders": 1000,
             "ticks": 600,
             "std_fundamental": 0.053,
             "init_assets": 740,
             'spread_max': 0.004,
             'money_multiplier': 2.2,
             "horizon": 200,
             "std_noise": 0.049,
             "w_random": 0.08,
             "strat_share_chartists": 0.08,
             "base_risk_aversion": 1.051,
             "fundamentalist_horizon_multiplier": 3.8,
             "trades_per_tick": 1, "mutation_intensity": 0.0477,
             "average_learning_ability": 0.05,
             "bond_mean_reversion": 0.0, 'cb_pf_range': 0.05,
             "qe_perc_size": 0.16, "cb_size": 0.02, "qe_asset_index": 0, "qe_start": 2, "qe_end":598}
        params.update(uncertain_parameters)

        list_of_seeds_params = [[seed, params] for seed in list_of_seeds]

        #TODO comment out
        #simulate_a_seed([0, params])

        costs = p.map(simulate_a_seed, list_of_seeds_params) # first argument is function to execute, second argument is tuple of all inputs TODO uncomment this

        return np.mean(costs)

    #TODO comment this out
    #model_performance(init_parameters)

    output = constrNM(model_performance, init_parameters, LB, UB, maxiter=10, full_output=True)

    with open('estimated_params.json', 'w') as f:
        json.dump(list(output['xopt']), f)

    print('All outputs are: ', output)


if __name__ == '__main__':
    pool_handler()
    print("The simulations took", time.time() - start_time, "to run")
