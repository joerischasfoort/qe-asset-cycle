from qe_model import *
from init_objects import *
from functions.helpers import *


def simulate_params_efast(NRUNS, parameter_set, fixed_parameters):
    """
    Simulate the model twice for different parameter sets. Once with BLR and once without. Record the difference in volatility.
    :param NRUNS: integer amount of Monte Carlo simulations
    :param parameter_set: list of parameters which have been sampled for Sobol sensitivity analysis
    :param fixed_parameters: list of parameters which will remain fixed
    :return: numpy array of average stylized facts outcome values for all parameter combinations
    """
    av_diff_BLR = []

    for parameters in parameter_set:
        # combine individual parameters with fixed parameters
        params = fixed_parameters.copy()
        params.update(parameters)

        scenarios = [None, 'BLR']

        # simulate the model

        av_volatility = {}
        for scenario in scenarios:
            trdrs = []
            orbs = []
            central_banks = []

            prices = []
            fundamentals = []
            for seed_nb in range(NRUNS):
                traders_nb, central_bank_nb, orderbook_nb = init_objects(params, seed_nb)
                traders_nb, central_bank_nb, orderbook_nb = qe_model(traders_nb, central_bank_nb, orderbook_nb,
                                                                     params, scenario=scenario, seed=seed_nb)
                central_banks.append(central_bank_nb)
                trdrs.append(traders_nb)
                orbs.append(orderbook_nb)

            prices = pd.DataFrame([orbs[run].tick_close_price for run in range(NRUNS)]).transpose()
            fundamentals = pd.DataFrame([orbs[run].fundamental for run in range(NRUNS)]).transpose()
            pfs = (prices / fundamentals)[:-1]
            # calculate volatility
            # detrend serie
            #stock_cycle, stock_trend = sm.tsa.filters.hpfilter(pfs, lamb=100000000)

            av_volatility[scenario] = np.mean(pfs.std()) / np.mean(pfs)

        av_diff_BLR.append(np.mean(av_volatility[scenarios[1]] - av_volatility[scenarios[0]]))

    return av_diff_BLR

# all_parameters = [{'std_noise': 0.08770628189875272,
#   'w_random': 0.041863280762066316,
#   'strat_share_chartists': 0.3214240696942189,
#   'base_risk_aversion': 0.8559495496103791,
#   'fundamentalist_horizon_multiplier': 3.933972354170189,
#   'mutation_intensity': 0.311810716621035,
#   'average_learning_ability': 0.6986081297643648}]
# #
# fixed_parameters = {"fundamental_value": 166,
#              "trader_sample_size": 22,
#              "n_traders": 1000,
#              "ticks": 500,
#              "std_fundamental": 0.053,
#              "init_assets": 740,
#              'spread_max': 0.004,
#              'money_multiplier': 2.2,
#              "horizon": 200,
#              "trades_per_tick": 1,
#              "bond_mean_reversion": 0.0, 'cb_pf_range': 0.05,
#              "qe_perc_size": 0.16, "cb_size": 0.024, "qe_asset_index": 0, "qe_start": 0, "qe_end":0}
# #
# print(simulate_params_efast(NRUNS=2, parameter_set=all_parameters, fixed_parameters=fixed_parameters))