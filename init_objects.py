from objects.trader import *
from objects.orderbook import *
from objects.exogenous_agents import *
import random
import numpy as np
from functions.helpers import calculate_covariance_matrix, div0


def init_objects(parameters, seed):
    """
    Init object for the distribution version of the model
    :param parameters:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    traders = []
    n_traders = parameters["n_traders"]

    weight_f = (1 - parameters['strat_share_chartists']) * (1 - parameters['w_random'])
    weight_c = parameters['strat_share_chartists'] * (1 - parameters['w_random'])

    f_points = int(weight_f * 100 * n_traders)
    c_points = int(weight_c * 100 * n_traders)
    r_points = int(parameters['w_random'] * 100 * n_traders)

    # create list of strategy points, shuffle it and divide in equal parts
    strat_points = ['f' for f in range(f_points)] + ['c' for c in range(c_points)] + ['r' for r in range(r_points)]
    random.shuffle(strat_points)
    agent_points = np.array_split(strat_points, n_traders)

    max_horizon = parameters['horizon'] * 2  # this is the max horizon of an agent if 100% fundamentalist
    historical_stock_returns = list(np.random.normal(0, parameters["std_fundamental"], max_horizon))

    total_stocks = 0

    for idx in range(n_traders):
        weight_fundamentalist = list(agent_points[idx]).count('f') / float(len(agent_points[idx]))
        weight_chartist = list(agent_points[idx]).count('c') / float(len(agent_points[idx]))
        weight_random = list(agent_points[idx]).count('r') / float(len(agent_points[idx]))

        init_stocks = int(np.random.uniform(0, parameters["init_assets"]))
        total_stocks += init_stocks
        init_money = np.random.uniform(0, (init_stocks * parameters['fundamental_value'] * parameters['money_multiplier']))

        c_share_strat = div0(weight_chartist, (weight_fundamentalist + weight_chartist))

        # initialize co_variance_matrix
        init_covariance_matrix = calculate_covariance_matrix(historical_stock_returns)
        init_active_orders = []

        lft_vars = TraderVariables(weight_fundamentalist, weight_chartist, weight_random, c_share_strat,
                                   init_money, init_stocks, init_covariance_matrix,
                                   parameters['fundamental_value'], init_active_orders)

        # determine heterogeneous horizon and risk aversion
        individual_horizon = np.random.randint(10, parameters['horizon'])

        individual_risk_aversion = abs(np.random.normal(parameters["base_risk_aversion"], parameters["base_risk_aversion"] / 5.0))
        individual_learning_ability = min(abs(np.random.normal(parameters['average_learning_ability'], 0.1)), 1.0)

        lft_params = TraderParameters(individual_horizon, individual_risk_aversion,
                                      individual_learning_ability, parameters['spread_max'])

        exp_returns = {'risky_asset': 0.0, 'money': 0.0}
        lft_expectations = TraderExpectations(parameters['fundamental_value'], exp_returns)
        traders.append(Trader(idx, lft_vars, lft_params, lft_expectations))

    # initialize central bank with assets at target
    asset_target_cb = int(parameters["qe_perc_size"] * total_stocks)
    asset_target = [asset_target_cb for t in range(parameters['ticks'])]

    ## TODO new
    # Determine QE volume
    if parameters["qe_end"] - parameters["qe_start"] > 0:
        QE_periods = parameters["qe_end"] - parameters["qe_start"]
        total_QE_volume = parameters["qe_perc_size"] * total_stocks
        period_volume = int(total_QE_volume / QE_periods)

        #asset_target = [0 for t in range(parameters['ticks'])]
        for t in range(parameters['ticks']):
            if t in range(parameters["qe_start"], parameters["qe_end"]):
                asset_target[t] = asset_target[t - 1] + period_volume
            elif t >= parameters["qe_end"]:
                asset_target[t] = asset_target[t - 1]
        ##

    cb_assets = [asset_target_cb for t in range(parameters['ticks'])]

    currency = np.zeros(parameters["ticks"])
    currency -= np.array(cb_assets) * parameters["fundamental_value"]

    asset_demand = 0

    init_active_orders_cb = []

    cb_pars = CBParameters(0.0)
    cb_vars = CBVariables(cb_assets, currency, asset_demand, asset_target, init_active_orders_cb)

    central_bank = CentralBank(cb_vars, cb_pars)

    # initialize the order book
    order_book = LimitOrderBook(parameters['fundamental_value'], parameters["std_fundamental"],
                                    max_horizon, parameters['ticks'])
    order_book.returns = list(historical_stock_returns)
    order_book.qe_period = [False for t in range(parameters['ticks'])]
    order_book.qt_period = [False for t in range(parameters['ticks'])]

    return traders, central_bank, order_book
