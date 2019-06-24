import random
import numpy as np
from functions.portfolio_optimization import *
from functions.helpers import calculate_covariance_matrix, div0, ornstein_uhlenbeck_evolve, npv


def qe_ineq_model(traders, central_bank, orderbooks, parameters, seed=1):
    """
    The main model function of distribution model where trader stocks are tracked.
    :param traders: list of Agent objects
    :param orderbook: object Order book
    :param parameters: dictionary of parameters
    :param seed: integer seed to initialise the random number generators
    :return: list of simulated Agent objects, object simulated Order book
    """
    random.seed(seed)
    np.random.seed(seed)
    assets = [parameters["asset_types"][a] + str(a) for a in range(len(parameters["fundamental_values"]))]
    simtime = (parameters["ticks"] + parameters['horizon'] + 1)

    # TODO new, possible remove if it does not work
    cash_flows = [[val / 100.0] for val in parameters["fundamental_values"]]
    # evolve cash flows fundamental value via random walk process
    for t in range(parameters["ticks"] * 2):
        for i, c in enumerate(cash_flows):
            if 'bond' in assets[i]:
                c.append(max(
                    ornstein_uhlenbeck_evolve(parameters["fundamental_values"][i] / 100.0, c[-1], parameters["std_fundamentals"][i],
                                              parameters['bond_mean_reversion'], seed), 0.1))
            else:
                c.append(max(c[-1] + parameters["std_fundamentals"][i] * np.random.randn(), 0.1))

    #f_val_cash_flows = [[npv(cash_flows[i][t:t + parameters["ticks"]], 0.01) for t in range(parameters["ticks"])] for i in range(len(cash_flows))]
    fundamentals = [[npv(cash_flows[i][:parameters["ticks"]], 0.01)] for i in range(len(cash_flows))]

    #fundamentals = [[val] for val in parameters["fundamental_values"]]

    for idx, ob in enumerate(orderbooks):
        ob.tick_close_price.append(fundamentals[idx][-1])

    traders_by_wealth = [t for t in traders]

    for tick in range(parameters['horizon'] + 1, parameters["ticks"] + parameters['horizon'] + 1):
        qe_tick = tick - parameters['horizon'] - 1  # correcting for the memory period necessary for traders

        if tick == parameters['horizon'] + 1:
            print('Start of simulation ', seed)

        print(tick)

        # update money and stocks history for agents
        for trader in traders:
            trader.var.money.append(trader.var.money[-1])
            asset_wealth = []
            for i in range(len(assets)):
                trader.var.assets[i].append(trader.var.assets[i][-1])
                asset_wealth.append(trader.var.assets[i][-1] * orderbooks[i].tick_close_price[-1])
            trader.var.wealth.append(trader.var.money[-1] + sum(asset_wealth))
            trader.var.weight_fundamentalist.append(trader.var.weight_fundamentalist[-1])
            trader.var.weight_chartist.append(trader.var.weight_chartist[-1])
            trader.var.weight_random.append(trader.var.weight_random[-1])

        # update money and assets for central bank
        central_bank.var.assets[parameters["qe_asset_index"]][qe_tick] = central_bank.var.assets[parameters["qe_asset_index"]][qe_tick - 1]
        central_bank.var.currency[qe_tick] = central_bank.var.currency[qe_tick - 1]

        # sort the traders by wealth to
        traders_by_wealth.sort(key=lambda x: x.var.wealth[-1], reverse=True)

        # evolve the fundamental value via random walk process
        for i, f in enumerate(fundamentals):
            if 'bond' in assets[i]:
                print('error bond not supported yet')
                #f.append(max(ornstein_uhlenbeck_evolve(parameters["fundamental_values"][i], f[-1], parameters["std_fundamentals"][i], parameters['bond_mean_reversion'], seed), 0.1))
            else:
                #f.append(max(f[-1] + parameters["std_fundamentals"][i] * np.random.randn(), 0.1))
                f.append(max(npv(cash_flows[i][qe_tick:qe_tick + parameters["ticks"]], 0.01), 0.1)) #TODO insert params risk free rate

        # allow for multiple trades in one day
        for turn in range(parameters["trades_per_tick"]):
            # Allow the central bank to do Quantitative Easing ####################################################
            #if qe_tick in range(parameters["qe_start"], parameters["qe_end"]):
            # Cancel any active orders
            for i, ob in enumerate(orderbooks):
                if central_bank.var.active_orders[i]:
                    for order in central_bank.var.active_orders[i]:
                        ob.cancel_order(order)
                        central_bank.var.active_orders[i] = []

            # determine demand
            cb_demand = int(central_bank.var.asset_target[qe_tick] - central_bank.var.assets[parameters["qe_asset_index"]][qe_tick])

            # Submit QE orders:
            if cb_demand > 0:
                bid = orderbooks[parameters["qe_asset_index"]].add_bid(orderbooks[parameters["qe_asset_index"]].lowest_ask_price, cb_demand, central_bank)
                central_bank.var.active_orders[parameters["qe_asset_index"]].append(bid)
            elif cb_demand < 0:
                ask = orderbooks[parameters["qe_asset_index"]].add_ask(orderbooks[parameters["qe_asset_index"]].highest_bid_price, -cb_demand, central_bank)
                central_bank.var.active_orders[parameters["qe_asset_index"]].append(ask)

            # END QE ##############################################################################################

            # select random sample of active traders
            active_traders = random.sample(traders, int((parameters['trader_sample_size'])))

            mid_prices = [np.mean([ob.highest_bid_price, ob.lowest_ask_price]) for ob in orderbooks]
            fundamental_components = [np.log(fundamentals[i][-1] / mid_prices[i]) for i in range(len(assets))]

            for i, ob in enumerate(orderbooks):
                ob.returns[-1] = (mid_prices[i] - ob.tick_close_price[-2]) / ob.tick_close_price[-2]

            chartist_components = [np.cumsum(ob.returns[:-len(ob.returns) - 1:-1]
                                             ) / np.arange(1., float(len(ob.returns) + 1)) for ob in orderbooks]

            for trader in active_traders:
                # Cancel any active orders
                for i, ob in enumerate(orderbooks):
                    if trader.var.active_orders[i]:
                        for order in trader.var.active_orders[i]:
                            ob.cancel_order(order)
                        trader.var.active_orders[i] = []

                def evolve(probability):
                    return random.random() < probability

                # Evolve an expectations parameter by learning from a successful trader or mutate at random
                if evolve(trader.par.learning_ability):
                    wealthy_trader = traders_by_wealth[random.randint(0, parameters['trader_sample_size'])]
                    trader.var.c_share_strat = np.mean([trader.var.c_share_strat, wealthy_trader.var.c_share_strat])
                else:
                    trader.var.c_share_strat = min(max(trader.var.c_share_strat * (1 + parameters['mutation_intensity'] * np.random.randn()), 0.01), 0.99)

                # update fundamentalist & chartist weights
                total_strat_weight = trader.var.weight_fundamentalist[-1] + trader.var.weight_chartist[-1]
                trader.var.weight_chartist[-1] = trader.var.c_share_strat * total_strat_weight
                trader.var.weight_fundamentalist[-1] = (1 - trader.var.c_share_strat) * total_strat_weight

                # record sentiment in orderbooks
                for ob in orderbooks:
                    ob.sentiment.append(np.array([trader.var.weight_fundamentalist[-1],
                                                  trader.var.weight_chartist[-1],
                                                  trader.var.weight_random[-1]]))

                # Update trader specific expectations
                noise_components = [parameters['std_noise'] * np.random.randn() for a in assets]

                # Expectation formation
                fcast_prices = []
                for i, a in enumerate(assets): #TODO fix below exp returns per stock asset
                    trader.exp.returns[a] = (
                        trader.var.weight_fundamentalist[-1] * np.divide(1, float(trader.par.horizon) * parameters["fundamentalist_horizon_multiplier"]) * fundamental_components[i] +
                        trader.var.weight_chartist[-1] * chartist_components[i][trader.par.horizon - 1] +
                        trader.var.weight_random[-1] * noise_components[i])
                    fcast_prices.append(mid_prices[i] * np.exp(trader.exp.returns[a]))

                observed_returns = []
                for ob in orderbooks:
                    obs_rets = ob.returns[-trader.par.horizon:]
                    if sum(np.abs(obs_rets)) > 0:
                        observed_returns.append(obs_rets)
                    else:
                        observed_returns.append(np.diff(ob.fundamental)[-trader.par.horizon:])

                trader.var.covariance_matrix = calculate_covariance_matrix(observed_returns, parameters) #TODo debug, does this work as intended?

                # employ portfolio optimization algo TODO if observed returns 0 (replace with stdev fundamental?
                ideal_trader_weights = portfolio_optimization(trader, tick) #TODO debug, does this still work as intended

                # Determine price and volume
                for i, ob in enumerate(orderbooks):
                    trader_price = np.random.normal(fcast_prices[i], trader.par.spread)
                    position_change = (ideal_trader_weights[assets[i]] * (
                                trader.var.assets[i][-1] * trader_price + trader.var.money[-1])) - (
                                                  trader.var.assets[i][-1] * trader_price)
                    volume = int(div0(position_change, trader_price))

                    # Trade:
                    if volume > 0:
                        volume = min(volume, int(div0(trader.var.money[-1], trader_price))) # the trader can only buy new stocks with money
                        bid = ob.add_bid(trader_price, volume, trader)
                        trader.var.active_orders[i].append(bid)
                    elif volume < 0:
                        volume = max(volume, -trader.var.assets[i][-1])  # the trader can only sell as much assets as it owns
                        ask = ob.add_ask(trader_price, -volume, trader)
                        trader.var.active_orders[i].append(ask)

            # Match orders in the order-book
            for i, ob in enumerate(orderbooks):
                while True:
                    matched_orders = ob.match_orders()
                    if matched_orders is None:
                        break
                    # execute trade
                    matched_orders[3].owner.sell(matched_orders[1], matched_orders[0] * matched_orders[1], i, qe_tick)
                    matched_orders[2].owner.buy(matched_orders[1], matched_orders[0] * matched_orders[1], i, qe_tick)

        for i, ob in enumerate(orderbooks):
            # Clear and update order-book history
            ob.cleanse_book()
            ob.fundamental = fundamentals[i]

    return traders, central_bank, orderbooks


def qe_ineq_active_cb_model(traders, central_bank, orderbooks, parameters, seed=1):
    """
    The main model function of distribution model where trader stocks are tracked.
    :param traders: list of Agent objects
    :param orderbook: object Order book
    :param parameters: dictionary of parameters
    :param seed: integer seed to initialise the random number generators
    :return: list of simulated Agent objects, object simulated Order book
    """
    random.seed(seed)
    np.random.seed(seed)
    assets = [parameters["asset_types"][a] + str(a) for a in range(len(parameters["fundamental_values"]))]

    fundamentals = [[val] for val in parameters["fundamental_values"]]

    for idx, ob in enumerate(orderbooks):
        ob.tick_close_price.append(fundamentals[idx][-1])

    traders_by_wealth = [t for t in traders]

    for tick in range(parameters['horizon'] + 1, parameters["ticks"] + parameters['horizon'] + 1):
        qe_tick = tick - parameters['horizon'] - 1  # correcting for the memory period necessary for traders

        if tick == parameters['horizon'] + 1:
            print('Start of simulation ', seed)

        print(tick)

        # update money and stocks history for agents
        for trader in traders:
            trader.var.money.append(trader.var.money[-1])
            asset_wealth = []
            for i in range(len(assets)):
                trader.var.assets[i].append(trader.var.assets[i][-1])
                asset_wealth.append(trader.var.assets[i][-1] * orderbooks[i].tick_close_price[-1])
            trader.var.wealth.append(trader.var.money[-1] + sum(asset_wealth))
            trader.var.weight_fundamentalist.append(trader.var.weight_fundamentalist[-1])
            trader.var.weight_chartist.append(trader.var.weight_chartist[-1])
            trader.var.weight_random.append(trader.var.weight_random[-1])

        # update money and assets for central bank
        central_bank.var.assets[parameters["qe_asset_index"]][qe_tick] = central_bank.var.assets[parameters["qe_asset_index"]][qe_tick - 1]
        central_bank.var.asset_target[qe_tick] = central_bank.var.asset_target[qe_tick - 1]
        central_bank.var.currency[qe_tick] = central_bank.var.currency[qe_tick - 1]

        # sort the traders by wealth to
        traders_by_wealth.sort(key=lambda x: x.var.wealth[-1], reverse=True)

        # evolve the fundamental value via random walk process
        for i, f in enumerate(fundamentals):
            if 'bond' in assets[i]:
                f.append(max(ornstein_uhlenbeck_evolve(parameters["fundamental_values"][i], f[-1], parameters["std_fundamentals"][i], parameters['bond_mean_reversion'], seed), 0.1))
            else:
                f.append(max(f[-1] + parameters["std_fundamentals"][i] * np.random.randn(), 0.1))

        # allow for multiple trades in one day
        for turn in range(parameters["trades_per_tick"]):
            # Allow the central bank to do Quantitative Easing ####################################################
            for i, ob in enumerate(orderbooks):
                if central_bank.var.active_orders[i]:
                    for order in central_bank.var.active_orders[i]:
                        ob.cancel_order(order)
                        central_bank.var.active_orders[i] = []

            # calculate PF ratio
            mid_prices = [np.mean([ob.highest_bid_price, ob.lowest_ask_price]) for ob in orderbooks]

            pf = mid_prices[parameters["qe_asset_index"]] / fundamentals[parameters["qe_asset_index"]][-1]
            # if pf is too high, decrease asset target otherwise increase asset target
            quantity_available = int(central_bank.var.assets[parameters["qe_asset_index"]][qe_tick] * parameters['cb_size'])
            if pf > 1 + parameters['cb_pf_range']: #  and central_bank.var.assets[parameters["qe_asset_index"]][qe_tick] >= parameters['cb_size']
                # sell assets
                central_bank.var.asset_target[qe_tick] -= quantity_available
                central_bank.var.asset_target[qe_tick] = max(central_bank.var.asset_target[qe_tick], 0)
            elif pf < 1 - parameters['cb_pf_range']:
                # buy assets
                central_bank.var.asset_target[qe_tick] += quantity_available

            # determine demand
            cb_demand = int(
                central_bank.var.asset_target[qe_tick] - central_bank.var.assets[parameters["qe_asset_index"]][qe_tick])

            # Submit QE orders:
            if cb_demand > 0:
                bid = orderbooks[parameters["qe_asset_index"]].add_bid(orderbooks[parameters["qe_asset_index"]].lowest_ask_price, cb_demand, central_bank)
                central_bank.var.active_orders[parameters["qe_asset_index"]].append(bid)
            elif cb_demand < 0:
                ask = orderbooks[parameters["qe_asset_index"]].add_ask(orderbooks[parameters["qe_asset_index"]].highest_bid_price, -cb_demand, central_bank)
                central_bank.var.active_orders[parameters["qe_asset_index"]].append(ask)

            # END QE ##############################################################################################
            #mid_prices = [np.mean([ob.highest_bid_price, ob.lowest_ask_price]) for ob in orderbooks]
            # select random sample of active traders
            active_traders = random.sample(traders, int((parameters['trader_sample_size'])))

            for i, ob in enumerate(orderbooks):
                ob.returns[-1] = (mid_prices[i] - ob.tick_close_price[-2]) / ob.tick_close_price[-2]

            fundamental_components = [np.log(fundamentals[i][-1] / mid_prices[i]) for i in range(len(assets))]
            chartist_components = [np.cumsum(ob.returns[:-len(ob.returns) - 1:-1]
                                             ) / np.arange(1., float(len(ob.returns) + 1)) for ob in orderbooks]

            for trader in active_traders:
                # Cancel any active orders
                for i, ob in enumerate(orderbooks):
                    if trader.var.active_orders[i]:
                        for order in trader.var.active_orders[i]:
                            ob.cancel_order(order)
                        trader.var.active_orders[i] = []

                def evolve(probability):
                    return random.random() < probability

                # Evolve an expectations parameter by learning from a successful trader or mutate at random
                if evolve(trader.par.learning_ability):
                    wealthy_trader = traders_by_wealth[random.randint(0, parameters['trader_sample_size'])]
                    trader.var.c_share_strat = np.mean([trader.var.c_share_strat, wealthy_trader.var.c_share_strat])
                else:
                    trader.var.c_share_strat = min(max(trader.var.c_share_strat * (1 + parameters['mutation_intensity'] * np.random.randn()), 0.01), 0.99)

                # update fundamentalist & chartist weights
                total_strat_weight = trader.var.weight_fundamentalist[-1] + trader.var.weight_chartist[-1]
                trader.var.weight_chartist[-1] = trader.var.c_share_strat * total_strat_weight
                trader.var.weight_fundamentalist[-1] = (1 - trader.var.c_share_strat) * total_strat_weight

                # record sentiment in orderbooks
                for ob in orderbooks:
                    ob.sentiment.append(np.array([trader.var.weight_fundamentalist[-1],
                                                  trader.var.weight_chartist[-1],
                                                  trader.var.weight_random[-1]]))

                # Update trader specific expectations
                noise_components = [parameters['std_noise'] * np.random.randn() for a in assets]

                # Expectation formation
                fcast_prices = []
                for i, a in enumerate(assets): #TODO fix below exp returns per stock asset
                    trader.exp.returns[a] = (
                        trader.var.weight_fundamentalist[-1] * np.divide(1, float(trader.par.horizon) * parameters["fundamentalist_horizon_multiplier"]) * fundamental_components[i] +
                        trader.var.weight_chartist[-1] * chartist_components[i][trader.par.horizon - 1] +
                        trader.var.weight_random[-1] * noise_components[i])
                    fcast_prices.append(mid_prices[i] * np.exp(trader.exp.returns[a]))

                observed_returns = []
                for ob in orderbooks:
                    obs_rets = ob.returns[-trader.par.horizon:]
                    if sum(np.abs(obs_rets)) > 0:
                        observed_returns.append(obs_rets)
                    else:
                        observed_returns.append(np.diff(ob.fundamental)[-trader.par.horizon:])

                trader.var.covariance_matrix = calculate_covariance_matrix(observed_returns, parameters) #TODo debug, does this work as intended?

                # employ portfolio optimization algo TODO if observed returns 0 (replace with stdev fundamental?
                ideal_trader_weights = portfolio_optimization(trader, tick) #TODO debug, does this still work as intended

                # Determine price and volume
                for i, ob in enumerate(orderbooks):
                    trader_price = np.random.normal(fcast_prices[i], trader.par.spread)
                    position_change = (ideal_trader_weights[assets[i]] * (
                                trader.var.assets[i][-1] * trader_price + trader.var.money[-1])) - (
                                                  trader.var.assets[i][-1] * trader_price)
                    volume = int(div0(position_change, trader_price))

                    # Trade:
                    if volume > 0:
                        volume = min(volume, int(div0(trader.var.money[-1], trader_price))) # the trader can only buy new stocks with money
                        bid = ob.add_bid(trader_price, volume, trader)
                        trader.var.active_orders[i].append(bid)
                    elif volume < 0:
                        volume = max(volume, -trader.var.assets[i][-1])  # the trader can only sell as much assets as it owns
                        ask = ob.add_ask(trader_price, -volume, trader)
                        trader.var.active_orders[i].append(ask)

            # Match orders in the order-book
            for i, ob in enumerate(orderbooks):
                while True:
                    matched_orders = ob.match_orders()
                    if matched_orders is None:
                        break
                    # execute trade
                    matched_orders[3].owner.sell(matched_orders[1], matched_orders[0] * matched_orders[1], i, qe_tick)
                    matched_orders[2].owner.buy(matched_orders[1], matched_orders[0] * matched_orders[1], i, qe_tick)

        for i, ob in enumerate(orderbooks):
            # Clear and update order-book history
            ob.cleanse_book()
            ob.fundamental = fundamentals[i]

    return traders, central_bank, orderbooks

