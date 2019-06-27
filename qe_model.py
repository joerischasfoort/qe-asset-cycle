import random
import numpy as np
from functions.portfolio_optimization import *
from functions.helpers import calculate_covariance_matrix, div0, ornstein_uhlenbeck_evolve, npv


def qe_model(traders, central_bank, orderbook, parameters, scenario=None, seed=1):
    """
    The main model function of distribution model where trader stocks are tracked.
    :param traders: list of Agent objects
    :param orderbook: object Order book
    :param parameters: dictionary of parameters
    :param scenario: can be the following strings: BUSTQE, BUSTQT, BOOMQE, BOOMQT
    :param seed: integer seed to initialise the random number generators
    :return: list of simulated Agent objects, object simulated Order book
    """
    random.seed(seed)
    np.random.seed(seed)

    fundamentals = [parameters["fundamental_value"]]

    orderbook.tick_close_price.append(fundamentals[-1])

    traders_by_wealth = [t for t in traders]

    for tick in range(parameters['horizon'] + 1, parameters["ticks"] + parameters['horizon'] + 1):
        qe_tick = tick - parameters['horizon'] - 1  # correcting for the memory period necessary for traders

        if tick == parameters['horizon'] + 1:
            print('Start of simulation ', seed)

        print(tick)
        if tick == 419:
            print('debug')

        # update money and stocks history for agents
        for trader in traders:
            trader.var.money.append(trader.var.money[-1])
            trader.var.assets.append(trader.var.assets[-1]) #TODO debug
            trader.var.wealth.append(trader.var.money[-1] + trader.var.assets[-1] * orderbook.tick_close_price[-1])
            trader.var.weight_fundamentalist.append(trader.var.weight_fundamentalist[-1])
            trader.var.weight_chartist.append(trader.var.weight_chartist[-1])
            trader.var.weight_random.append(trader.var.weight_random[-1])

        # update money and assets for central bank
        central_bank.var.assets[qe_tick] = central_bank.var.assets[qe_tick - 1]
        central_bank.var.asset_target[qe_tick] = central_bank.var.asset_target[qe_tick - 1]
        central_bank.var.currency[qe_tick] = central_bank.var.currency[qe_tick - 1]

        # sort the traders by wealth to
        traders_by_wealth.sort(key=lambda x: x.var.wealth[-1], reverse=True)

        # evolve the fundamental value via random walk process
        if True: #TODO decide what to use.
            fundamentals.append(max(ornstein_uhlenbeck_evolve(parameters["fundamental_value"], fundamentals[-1], parameters["std_fundamental"], parameters['bond_mean_reversion'], seed), 0.1))
        else:
            fundamentals.append(max(fundamentals[-1] + parameters["std_fundamentals"] * np.random.randn(), 0.1))

        # allow for multiple trades in one day
        for turn in range(parameters["trades_per_tick"]):
            # Allow the central bank to do Quantitative Easing ####################################################
            if central_bank.var.active_orders:
                for order in central_bank.var.active_orders:
                    orderbook.cancel_order(order)
                    central_bank.var.active_orders = []

            # calculate PF ratio
            mid_price = np.mean([orderbook.highest_bid_price, orderbook.lowest_ask_price])

            pf = mid_price / fundamentals[-1]
            # if pf is too high, decrease asset target otherwise increase asset target
            quantity_available = int(central_bank.var.assets[qe_tick] * parameters['cb_size']) # TODO debug

            if scenario == 'BUSTQE':
                if pf < 1 - parameters['cb_pf_range']:
                    # buy assets
                    central_bank.var.asset_target[qe_tick] += quantity_available
            elif scenario == 'BUSTQT':
                if pf < 1 - parameters['cb_pf_range']:
                    # sell assets
                    central_bank.var.asset_target[qe_tick] -= quantity_available
                    central_bank.var.asset_target[qe_tick] = max(central_bank.var.asset_target[qe_tick], 0)

            elif scenario == 'BOOMQE':
                if pf > 1 + parameters['cb_pf_range']:
                    # buy assets
                    central_bank.var.asset_target[qe_tick] += quantity_available

            elif scenario == 'BOOMQT':
                if pf > 1 + parameters['cb_pf_range']:
                    # sell assets
                    central_bank.var.asset_target[qe_tick] -= quantity_available
                    central_bank.var.asset_target[qe_tick] = max(central_bank.var.asset_target[qe_tick], 0)

            elif scenario == 'BLR':
                if pf > 1 + parameters['cb_pf_range']:
                    # sell assets
                    central_bank.var.asset_target[qe_tick] -= quantity_available
                    central_bank.var.asset_target[qe_tick] = max(central_bank.var.asset_target[qe_tick], 0)
                elif pf < 1 - parameters['cb_pf_range']:
                    # buy assets
                    central_bank.var.asset_target[qe_tick] += quantity_available

            # determine demand
            cb_demand = int(central_bank.var.asset_target[qe_tick] - central_bank.var.assets[qe_tick]) # TODO debug

            # Submit QE orders:
            if cb_demand > 0:
                bid = orderbook.add_bid(orderbook.lowest_ask_price, cb_demand, central_bank)
                central_bank.var.active_orders.append(bid)
                print('cb QE')
                orderbook.qe_period[qe_tick] = True
            elif cb_demand < 0:
                ask = orderbook.add_ask(orderbook.highest_bid_price, -cb_demand, central_bank)
                central_bank.var.active_orders.append(ask)
                print('cb QT')
                orderbook.qt_period[qe_tick] = True

            # END QE ##############################################################################################
            # select random sample of active traders
            active_traders = random.sample(traders, int((parameters['trader_sample_size'])))

            orderbook.returns[-1] = (mid_price - orderbook.tick_close_price[-2]) / orderbook.tick_close_price[-2] #todo is this correct

            fundamental_component = np.log(fundamentals[-1] / mid_price)
            chartist_component = np.cumsum(orderbook.returns[:-len(orderbook.returns) - 1:-1]
                                            ) / np.arange(1., float(len(orderbook.returns) + 1))

            for trader in active_traders:
                # Cancel any active orders
                if trader.var.active_orders:
                    for order in trader.var.active_orders:
                        orderbook.cancel_order(order)
                    trader.var.active_orders = []

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

                # record sentiment in orderbook
                orderbook.sentiment.append(np.array([trader.var.weight_fundamentalist[-1],
                                              trader.var.weight_chartist[-1],
                                              trader.var.weight_random[-1]]))

                # Update trader specific expectations
                noise_component = parameters['std_noise'] * np.random.randn()

                # Expectation formation
                #TODO fix below exp returns per stock asset
                trader.exp.returns['risky_asset'] = (
                    trader.var.weight_fundamentalist[-1] * np.divide(1, float(trader.par.horizon) * parameters["fundamentalist_horizon_multiplier"]) * fundamental_component +
                    trader.var.weight_chartist[-1] * chartist_component[trader.par.horizon - 1] +
                    trader.var.weight_random[-1] * noise_component)

                fcast_price = mid_price * np.exp(trader.exp.returns['risky_asset'])

                obs_rets = orderbook.returns[-trader.par.horizon:]
                if sum(np.abs(obs_rets)) > 0:
                    observed_returns = obs_rets
                else:
                    observed_returns = np.diff(fundamentals)[-trader.par.horizon:]

                trader.var.covariance_matrix = calculate_covariance_matrix(observed_returns) #TODo debug, does this work as intended?

                # employ portfolio optimization algo TODO if observed returns 0 (replace with stdev fundamental?
                ideal_trader_weights = portfolio_optimization(trader, tick) #TODO debug, does this still work as intended

                # Determine price and volume
                trader_price = np.random.normal(fcast_price, trader.par.spread)
                position_change = (ideal_trader_weights['risky_asset'] * (
                        trader.var.assets[-1] * trader_price + trader.var.money[-1])) - (
                                              trader.var.assets[-1] * trader_price)
                volume = int(div0(position_change, trader_price))

                # Trade:
                if volume > 0:
                    volume = min(volume, int(div0(trader.var.money[-1], trader_price))) # the trader can only buy new stocks with money
                    bid = orderbook.add_bid(trader_price, volume, trader)
                    trader.var.active_orders.append(bid)
                elif volume < 0:
                    volume = max(volume, -trader.var.assets[-1])  # the trader can only sell as much assets as it owns
                    ask = orderbook.add_ask(trader_price, -volume, trader)
                    trader.var.active_orders.append(ask)

            # Match orders in the order-book
            while True:
                matched_orders = orderbook.match_orders()
                if matched_orders is None:
                    break
                # execute trade
                matched_orders[3].owner.sell(matched_orders[1], matched_orders[0] * matched_orders[1], qe_tick)
                matched_orders[2].owner.buy(matched_orders[1], matched_orders[0] * matched_orders[1], qe_tick)

        # Clear and update order-book history
        orderbook.cleanse_book()
        orderbook.fundamental = fundamentals

    return traders, central_bank, orderbook

