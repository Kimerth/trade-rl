from typing import List
import pandas as pd
from pandas.core.frame import DataFrame
import tensortrade.env.default as default

from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USDT, Instrument
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.env.default.actions import SimpleOrders
from tensortrade.env.default.rewards import SimpleProfit

from os import listdir


def rsi(price: Stream[float], period: float) -> Stream[float]:
    r = price.diff()
    upside = r.clamp_min(0).abs()
    downside = r.clamp_max(0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return 100*(1 - (1 + rs) ** -1)


def macd(price: Stream[float], fast: float, slow: float, signal: float) -> Stream[float]:
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return signal

from tensortrade.env.generic import Stopper
class LossStopper(Stopper):
    def __init__(self, max_allowed_loss: float):
        super().__init__()
        self.max_allowed_loss = max_allowed_loss

        self._max_net_worth = 0

    def stop(self, env) -> bool:
        net_worth = env.action_scheme.portfolio.net_worth
        self._max_net_worth = max(net_worth, self._max_net_worth)
        loss = 1.0 - net_worth / self._max_net_worth
        c1 = loss > self.max_allowed_loss
        c2 = not env.observer.has_next()
        return c1 or c2

    def reset(self) -> None:
        self._max_net_worth = 0


def find_csv_filenames( path_to_dir, frame, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith(suffix) and frame in filename ]

path_to_data = 'data/'
all_csvs = find_csv_filenames(path_to_data, '1m')

coins = []
coin_names = []

for csv_name in all_csvs:
    coin_names.append(csv_name.split('-')[0].replace('USDT', ''))
    coins.append(Instrument(coin_names[-1], 8))

path_to_data = 'data/'
all_csvs = find_csv_filenames(path_to_data, '15m')
print(all_csvs)

coins = []
coin_names = []

for csv_name in all_csvs:
    coin_names.append(csv_name.split('-')[0].replace('USDT', ''))
    coins.append(Instrument(coin_names[-1], 8))

data: List[DataFrame] = []
for csv_name in all_csvs:
    data.append(pd.read_csv(path_to_data + csv_name))

from operator import itemgetter
min_coin_data_idx, _ = min(enumerate(map(len, data)), key=itemgetter(1))
min_coin_data_date = data[min_coin_data_idx]['date'][0]

def get_train_env(start_date: str = '2018', max_steps: int = None):
    masked_data: List[DataFrame] = [None] * len(data)
    for i in range(len(data)):
        mask = (data[i]['date'] >= min_coin_data_date) & (data[i]['date'] >= start_date)
        masked_data[i] = data[i].loc[mask]
        if max_steps:
            masked_data[i] = masked_data[i].iloc[:max_steps]

    features = []
    for idx, d in enumerate(masked_data):
        cp = Stream.source(list(d['close']), dtype='float')
        features += [
            cp.log().diff().rename("lr_%s" % coin_names[idx]),
            rsi(cp, period=48).rename("rsi_%s" % coin_names[idx]),
            macd(cp, fast=10, slow=50, signal=5).rename("macd_%s" % coin_names[idx])
        ]

    feed = DataFeed(features)
    feed.compile()

    exchange = Exchange("exchange", service=execute_order, options=ExchangeOptions(commission=0.001))(
        *[Stream.source(list(d["close"]), dtype="float").rename(f'USDT-{coin_names[idx]}') for idx, d in enumerate(masked_data)]
    )

    portfolio = Portfolio(USDT, [Wallet(exchange, 100 * USDT)] + [Wallet(exchange, 0 * x) for x in coins])
    env = default.create(
        portfolio=portfolio,
        action_scheme=SimpleOrders(trade_sizes=2, min_order_pct=0.2),
        reward_scheme=SimpleProfit(),
        feed=feed,
        renderer=default.renderers.EmptyRenderer(),
        window_size=96,
        stopper=LossStopper(0.5)
    )

    return env