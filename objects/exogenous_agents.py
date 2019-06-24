
class CentralBank:
    """
    Class holding central bank properties
    """

    def __init__(self, variables, parameters):
        self.var = variables
        self.par = parameters

    def __repr__(self):
        return 'central_bank_' + str(self.par.country)

    def sell(self, amount, price, asset_index, t):
        """
        Sells `amount` of stocks for a total of `price`
        :param amount: int Number of stocks sold.
        :param price: float Total price for stocks.
        :return: -
        """
        if self.var.assets[asset_index][-1] < amount:
            raise ValueError("not enough stocks to sell this amount")
        self.var.assets[asset_index][t] -= amount
        self.var.currency[t] += price

    def buy(self, amount, price, asset_index, t):
        """
        Buys `amount` of stocks for a total of `price`
        :param amount: int number of stocks bought.
        :param price: float total price for stocks.
        :return: -
        """
        # the central bank is not restricted by the stock of money to buy assets, it creates money.

        self.var.assets[asset_index][t] += amount
        self.var.currency[t] -= price


class CBVariables:
    """Holds the inital variables for the central bank"""
    def __init__(self, assets, currency, asset_demand, asset_target, active_orders):
        self.assets = assets # The quantities of assets held
        self.currency = currency # The quantities of currencies held
        self.asset_demand = [asset_demand]
        self.asset_target = asset_target
        self.active_orders = active_orders


class CBParameters:
    """Holds the inital variables for the central bank"""
    def __init__(self, reserve_rate):
        self.reserve_rate = reserve_rate

