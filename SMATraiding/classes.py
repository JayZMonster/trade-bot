from settings import sma_logger


class TransactionCostMoreThanBank(Exception):

    def __init__(self, req_bank):
        self._required_bank = req_bank

    def __str__(self):
        return "You've just set your transaction cost more than your start bank. Change it to the number that" \
               f"is less or equals {self._required_bank}"


class MoneyIsOver(Exception):

    def __str__(self):
        return 'You have not enough money in your bank to proceed this operation!'


class Bankroll:

    def __init__(self, start_bank):
        self._start_bank = start_bank
        self._current_bank = start_bank
        self._transaction_cost = self._start_bank
        self._crypto_bank = 0
        self._bought_cryptos = []

    def set_transaction_cost(self, price):
        self._transaction_cost = price
        if float(self._transaction_cost) > float(self._start_bank):
            raise TransactionCostMoreThanBank(self._start_bank)

    def get_transaction_cost(self):
        return self._transaction_cost

    def set_current_bank(self, value: int):
        self._current_bank = value

    def add_funds(self, funds):
        self._current_bank += funds

    def get_current_bank(self):
        return self._current_bank

    def get_crypto_bank(self):
        return self._crypto_bank

    def get_tokens_on_hold(self):
        return self._bought_cryptos

    @staticmethod
    def buy_alert(symbol, amount, cost):
        print(f'{amount} of {symbol} bought for {cost}!')
        sma_logger.info(f'{amount} of {symbol} bought for {cost}!')

    @staticmethod
    def sell_alert(symbol, amount, cost):
        print(f'{amount} of {symbol} sold for {cost}!')
        sma_logger.info(f'{amount} of {symbol} sold for {cost}!')

    def buy_crypto(self, crypto_price, symbol):
        self._current_bank -= self._transaction_cost
        if self._current_bank < 0:
            raise MoneyIsOver
        self._crypto_bank += float(self._transaction_cost)/float(crypto_price)
        self._bought_cryptos.append(symbol)
        self.buy_alert(symbol, self._crypto_bank, float(crypto_price))

    def sell_crypto(self, crypto_price, symbol):
        self._bought_cryptos.remove(symbol)
        self._current_bank += float(self._crypto_bank)*float(crypto_price)
        self.sell_alert(symbol, self._crypto_bank, float(self._crypto_bank)*float(crypto_price))
        self._crypto_bank = 0
