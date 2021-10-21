from binance.client import Client
import asyncio
from binance import AsyncClient, BinanceSocketManager
import pandas as pd  # needs pip install
import numpy as np

from get_gainers import get_hourly_dataframe
from settings import API_KEY, API_SECRET


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


def prepare_sma(symbol_df):
    """
    Defines logic of trading strategy by calculating SMA's
    :param symbol_df:
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

    return symbol_df


def calc_derivative(start_time, end_time, start_point, end_point):
    time_diff = end_time - start_time
    time_diff = np.array(time_diff).astype('timedelta64[m]')
    time_diff = time_diff.astype(float)/60.0
    price_diff = float(end_point) - float(start_point)
    return float(price_diff)/float(time_diff)


def derivatives(token_df, current):
    positive_positions = token_df[token_df['Position'] == 1.0]
    start_time, start_point = positive_positions.iloc[-1].Time, positive_positions.iloc[-1].Close

    # Getting required data to calculate a derivatives
    end_time, end_point = pd.to_datetime(int(current['E']), unit='ms'), float(current['p'])
    sub_time, sub_point = token_df.iloc[-1].Time,  token_df.iloc[-1].Close
    try:
        first_der_sub_point = calc_derivative(start_time, sub_time, start_point, sub_point)
    except ZeroDivisionError:
        first_der_sub_point = sub_point- start_point
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
        print('Calculation Error:(0)')
        return end_point - sub_point
    print(second_der_end_point)
    return second_der_end_point, end_point


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


async def crossover_logic(symbol):
    """
    Looking for top ten gainers, then taking their price's history, dropping currencies that does not match requirements
    :return:
    """
    client = Client(API_KEY[0], API_SECRET)
    hour_df_symbol = get_hourly_dataframe(symbol, client)
    sma_prepared_data = prepare_sma(hour_df_symbol)
    sma_prepared_data.reset_index(inplace=True)
    current_ticker = await get_curr_price(symbol)
    derivative, last_price = derivatives(sma_prepared_data, current_ticker)
    return derivative, last_price


# if __name__ == "__main__":
#     client = Client(API_KEY[0], API_SECRET, testnet=True)
#     print("Using Binance TestNet Server")
#     # loop = asyncio.get_event_loop()
#     # loop.run_until_complete(buy_or_sell(ticker))
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(crossover_logic('SOLUSDT'))
