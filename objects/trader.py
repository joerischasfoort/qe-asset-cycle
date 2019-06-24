import numpy as np


class Trader:
    """Class holding low frequency trader properties"""
    def __init__(self, name, variables, parameters, expectations):
        """
        Initialize trader class
        :param name: integer number which will be the name of the trader
        :param variables: object Tradervariables
        :param parameters: object TraderParameters
        :param expectations: object TraderExpectations
        """
        self.name = name
        self.var = variables
        self.par = parameters
        self.exp = expectations

    def __repr__(self):
        """
        :return: String representation of the trader
        """
        return 'Trader' + str(self.name)

    def sell(self, amount, price, asset_index, t=0):
        """
        Sells `amount` of stocks for a total of `price`
        :param amount: int Number of stocks sold.
        :param price: float Total price for stocks.
        :return: -
        """
        if self.var.assets[asset_index][-1] < amount:
            raise ValueError("not enough stocks to sell this amount")
        self.var.assets[asset_index][-1] -= amount
        self.var.money[-1] += price

    def buy(self, amount, price, asset_index, t=0):
        """
        Buys `amount` of stocks for a total of `price`
        :param amount: int number of stocks bought.
        :param price: float total price for stocks.
        :return: -
        """
        if self.var.money[-1] < price:
            raise ValueError("not enough money to buy this amount of stocks")

        self.var.assets[asset_index][-1] += amount
        self.var.money[-1] -= price


class TraderVariables:
    """
    Holds the initial variables for the traders
    """
    def __init__(self, weight_fundamentalist, weight_chartist, weight_random, c_share_strat,
                 money, assets, covariance_matrix, init_prices, active_orders):
        """
        Initializes variables for the trader
        :param weight_fundamentalist: float fundamentalist expectation component
        :param weight_chartist: float trend-following chartism expectation component
        :param weight_random: float random or heterogeneous expectation component
        :param weight_mean_reversion: float mean-reversion chartism expectation component
        """
        self.weight_fundamentalist = [weight_fundamentalist]
        self.weight_chartist = [weight_chartist]
        self.weight_random = [weight_random]
        self.c_share_strat = c_share_strat
        self.money = [money]
        self.assets = assets
        self.wealth = [money + sum(np.array([a[-1] for a in assets]) * np.array(init_prices))]
        self.covariance_matrix = covariance_matrix
        self.active_orders = active_orders


class TraderParameters:
    """
    Holds the the trader parameters for the distribution model
    """

    def __init__(self, ref_horizon, risk_aversion, learning_ability, max_spread):
        """
        Initializes trader parameters
        :param ref_horizon: integer horizon over which the trader can observe the past
        :param max_spread: Maximum spread at which the trader will submit orders to the book
        :param risk_aversion: float aversion to price volatility
        """
        self.horizon = ref_horizon
        self.risk_aversion = risk_aversion
        self.learning_ability = learning_ability
        self.spread = max_spread * np.random.rand()


class TraderExpectations:
    """
    Holds the agent expectations for several variables
    """
    def __init__(self, price, returns):
        """
        Initializes trader expectations
        :param price: float
        """
        self.price = price
        self.returns = returns

