#!python
import asyncio
import json

from binance import BinanceSocketManager, AsyncClient
from time import sleep

from settings import sma_logger
from get_gainers import get_gainers
from classes import Bankroll
from sma_crossover import crossover_logic


async def main(ticker):
    """
    Used for getting prices of required currencies
    :param ticker:
    :return:
    """
    try:
        client = await AsyncClient.create()
    except:
        sleep(1.5)
        client = await AsyncClient.create()
    bm = BinanceSocketManager(client)

    # start any sockets here, i.e a trade socket
    ts = bm.trade_socket(ticker)

    # then start receiving messages
    async with ts as tscm:
        res = await tscm.recv()
        return res

    await client.close_connection()


async def buy(tokens, total, bankrolls_list):
    print('Buying', flush=True)
    sma_logger.info("Buying")

    tokens_prices = []
    for token in tokens:
        try:
            tokens_prices.append(await main(token))
        except:
            sleep(2)
            tokens_prices.append(await main(token))
    print(tokens_prices)
    for token_price in tokens_prices:
        print(token_price)
        bank = Bankroll(start_bank=25)
        bank.set_transaction_cost(price=20)
        bank.buy_crypto(token_price['p'], token_price['s'])
        total.add_funds(-1*float(bank.get_transaction_cost()))
        bankrolls_list.append(bank)
        print('Successfully bought', flush=True)
        sma_logger.info("Successfully bought")

    return bankrolls_list


async def sell(bankrolls, total):
    idx_to_del = []
    start_len = len(bankrolls)
    for idx, bank in enumerate(bankrolls):
        symbol = bank.get_tokens_on_hold()[0]
        derivative, last_price = await crossover_logic(symbol)
        if float(derivative) < 0:
            total.add_funds(float(bank.get_crypto_bank())*float(last_price))
            bank.sell_crypto(last_price, symbol)

            idx_to_del.append(idx)

    for i in idx_to_del[-1::-1]:
        bankrolls.pop(i)

    if start_len == len(bankrolls):
        return [], False
    return [token.get_tokens_on_hold()[0] for token in bankrolls], True


async def buy_and_sell():
    """
    Implements strategy, trading function
    :param flag:
    :return:
    """
    bankrolls_list = []
    tokens_on_hold = []
    amount_of_tokens = 5
    total_bank = Bankroll(start_bank=0)
    print("Work starts", flush=True)
    sma_logger.info("Work starts")
    while True:
        print('Taking top gaining tokens', flush=True)
        sma_logger.info("\nTaking top gaining tokens")
        tokens = await get_gainers(amount=amount_of_tokens, existing_tokens=tokens_on_hold)
        print(tokens)
        if not tokens:
            print('No tokens to buy! Will sleep for 10 minutes!', flush=True)
            sma_logger.info("No tokens to buy! Will sleep for 10 minutes!")
            sleep(600)
            continue
        print('These tokens are:', tokens, flush=True)
        sma_logger.info(f"These tokens are: {tokens}")
        banks_list = await buy(tokens, total_bank, bankrolls_list)
        order_posted = False
        sma_logger.info(f"Sleeping for 20 minutes")
        sleep(1200)

        # Trying to sell
        print('Selling', flush=True)
        sma_logger.info(f"Selling")
        while not order_posted:
            tokens_on_hold, bool_value = await sell(banks_list, total_bank)
            order_posted = bool_value
            if order_posted is False:
                print("Not time yet to sell! Waiting 10 minutes more!", flush=True)
                sma_logger.info(f"Not time yet to sell! Waiting 10 minutes more!")
                sleep(600)
        print(f'------- TOTAL BANK FOR NOW: {total_bank.get_current_bank()} -------', flush=True)
        sma_logger.info(f"------- TOTAL BANK FOR NOW: {total_bank.get_current_bank()} -------")

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(buy_and_sell())
