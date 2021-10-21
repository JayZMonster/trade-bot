from time import sleep

from binance import AsyncClient, BinanceSocketManager, Client
from settings import API_KEY, API_SECRET
import pandas as pd  # needs pip install
import numpy as np
import asyncio


def create_frame(msg) -> pd.DataFrame:
    """
    Creates a pd Dataframe of history data, used for get_hourly_dataframe
    :param msg:
    :return pd.DataFrame:
    """
    frame = pd.DataFrame(data=np.array(msg).reshape(len(msg),-1))
    frame = frame.loc[:, :4]
    frame.columns = ['Time', 'Open', 'High', 'Low', 'Close']
    frame.Open = frame.Open.astype(float)
    frame.Low = frame.Low.astype(float)
    frame.Close = frame.Close.astype(float)
    frame.High = frame.High.astype(float)
    frame.Time = pd.to_datetime(frame.Time, unit='ms')
    frame.reset_index()
    frame.set_index('Time', inplace=True)
    return frame


def get_hourly_dataframe(symbol, client):
    # valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    # request historical candle (or klines) data using timestamp from above, interval either every min, hr, day or month
    # starttime = '30 minutes ago UTC' for last 30 mins time
    # e.g. client.get_historical_klines(symbol='ETHUSDTUSDT', '1m', starttime)
    # starttime = '1 Dec, 2017', '1 Jan, 2018'  for last month of 2017
    # e.g. client.get_historical_klines(symbol='BTCUSDT', '1h', "1 Dec, 2017", "1 Jan, 2018")
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "30 days ago UTC")
    frame = create_frame(klines)
    return frame


def get_history_for_gainers(top_ten, client):
    """
    Gets history data for each symbol of top ten gainers
    :param top_ten:
    :return:
    """
    top_fifteen_history = []
    for idx, sym in top_ten.iterrows():
        top_fifteen_history.append(get_hourly_dataframe(sym[0], client))
    return top_fifteen_history


def create_gainers_frame(msg):
    idx_to_del = []
    for idx, ticker in enumerate(msg):
        if ticker['s'][-4:] != 'USDT':
            idx_to_del.append(idx)
        elif ('UP' in ticker['s']) or ('DOWN' in ticker['s']):
            idx_to_del.append(idx)

    for i in idx_to_del[-1::-1]:
        msg.pop(i)
    frame = pd.DataFrame(data=msg)
    frame = frame.loc[:, ['s', 'c', 'P']]
    frame.columns = ['Symbol', 'Close', 'Price_change']
    frame.Close = frame.Close.astype(float)
    frame.Price_change = frame.Price_change.astype(float)
    frame = frame.sort_values(by='Price_change', ascending=False)
    return frame


def sma_logic(symbol_df):
    """
    Defines logic of trading strategy by calculating SMA's
    :param symbol:
    :return:
    """
    # small time Moving average. calculate 5 moving average using Pandas over close price
    symbol_df['7sma'] = symbol_df['Close'].rolling(7).mean()
    # long time moving average. calculate 15 moving average using Pandas
    symbol_df['25sma'] = symbol_df['Close'].rolling(25).mean()
    # Calculate signal column
    symbol_df['Signal'] = np.where(symbol_df['7sma'] > symbol_df['25sma'], 1, 0)
    # Calculate position column with diff
    symbol_df['Position'] = symbol_df['Signal'].diff()
    # get the column=Position as a list of items.
    low_sma = symbol_df['7sma'].tolist()
    high_sma = symbol_df['25sma'].tolist()
    position = symbol_df['Position'].tolist()
    return low_sma, high_sma, position, symbol_df


def calc_derivative(start_time, end_time, start_point, end_point):
    time_diff = end_time - start_time
    time_diff = np.array(time_diff).astype('timedelta64[m]')
    time_diff = time_diff.astype(float)/60.0
    price_diff = float(end_point) - float(start_point)
    return float(price_diff)/float(time_diff)


def derivatives(token_df, current):
    token_df.reset_index(inplace=True)
    positive_positions = token_df[token_df['Position'] == 1.0]
    print(positive_positions)
    start_time, start_point = positive_positions.iloc[-1].Time, positive_positions.iloc[-1].Close

    # Getting required data to calculate a derivatives
    end_time, end_point = pd.to_datetime(int(current['E']), unit='ms'), float(current['p'])
    sub_time, sub_point = token_df.iloc[-2].Time,  token_df.iloc[-2].Close
    try:
        first_der_sub_point = calc_derivative(start_time, sub_time, start_point, sub_point)
    except ZeroDivisionError:
        first_der_sub_point = sub_point - start_point
    print(f'First derivative(sub_point) {first_der_sub_point}')
    try:
        first_der_end_point = calc_derivative(start_time, end_time, start_point, end_point)
    except ZeroDivisionError:
        first_der_end_point = end_point - start_point
    print(f'First derivative(end_point) {first_der_end_point}')
    try:
        second_der_end_point = calc_derivative(start_time=sub_time,
                                               end_time=end_time,
                                               start_point=first_der_sub_point,
                                               end_point=first_der_end_point)
    except ZeroDivisionError:
        return end_point - sub_point
    print(second_der_end_point)
    return second_der_end_point, end_point > sub_point


async def get_curr_price(symbol):
    try:
        client = await AsyncClient.create()
    except:
        sleep(1.5)
        client = await AsyncClient.create()
    bm = BinanceSocketManager(client)

    # start any sockets here, i.e a trade socket
    ts = bm.trade_socket(symbol)

    # then start receiving messages
    async with ts as tscm:
        res = await tscm.recv()
        return res


async def confirm_gainers(history, frame):
    """
    Confirms tokens parameters: Current price > Last hour price
    :param history:
    :param frame:
    :return:
    """
    applied_tokens = []
    idx_to_del = []
    print(frame)
    symbols = frame.Symbol.tolist()
    for idx, token in enumerate(history):
        low_sma, high_sma, position, symbol_df = sma_logic(token)
        print(f'ID:{idx}, low sma:{low_sma[-1]}, high sma:{high_sma[-1]}, position1:{position[-1]}, position2:{position[-2]}, type:{type(position[-1])}')
        current_ticker = await get_curr_price(symbols[idx])
        current_price_der, compare_last_points = derivatives(symbol_df, current_ticker)

        if float(low_sma[-1]) < float(high_sma[-1]):
            if idx not in idx_to_del:
                idx_to_del.append(idx)

        if not compare_last_points:
            if current_price_der < 0:
                if idx not in idx_to_del:
                    idx_to_del.append(idx)

    for i in idx_to_del[-1::-1]:
        frame.drop(i, inplace=True)
        history.pop(i)

    current_prices = frame.Close.tolist()
    frame.reset_index(inplace=True, drop=True)
    last_hour_prices = [token.iloc[-2].Close for token in history]

    print(frame)
    for current_price, last_price in zip(current_prices, enumerate(last_hour_prices)):
        print(f'ID:{last_price[0]} \nCurrent: {current_price}, Last hour: {last_price[1]}')

        if float(current_price) > float(last_price[1]):
            print('Current:', current_price, '\nLast hour:', last_price[1])
            applied_tokens.append(frame.Symbol[last_price[0]])
    return applied_tokens


async def gainers_info():
    """
    Getting all tickers data for last 24hr
    :return:
    """
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    # start any sockets here, i.e a trade socket
    ts = bm.ticker_socket()
    async with ts as tscm:
        res = await tscm.recv()
        await client.close_connection()
        return res


async def get_gainers(amount, existing_tokens=None):
    """
    Main function that runs all the code in here
    :param amount:
    :param existing_tokens:
    :return:
    """
    client = Client(API_KEY[0], API_SECRET)
    raw_data = await gainers_info()
    frame = create_gainers_frame(raw_data)

    # Taking first 15 gainers
    frame = frame.head(15)
    frame.reset_index(inplace=True, drop=True)
    gainers_history = get_history_for_gainers(frame, client)
    applied_tokens = await confirm_gainers(gainers_history, frame)

    # Removing tokens that are not sold yet, to prevent buying single token multiple times
    try:
        for i in existing_tokens:
            if i in applied_tokens:
                applied_tokens.remove(i)
    except TypeError:
        print('No existing tokens or typeError')
        pass

    if len(applied_tokens) > int(amount):
        applied_tokens = applied_tokens[:amount]
    print(applied_tokens)
    return applied_tokens

# if __name__ == '__main__':
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(get_gainers(1, ['BTCUSDT', '']))
