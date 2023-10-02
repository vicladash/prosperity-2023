import pandas as pd
import matplotlib.pyplot as plt
import re
import io

raw = ''
with open('raw.log', 'r') as f:
    raw = f.read()

hdf = pd.read_csv(io.StringIO(raw[raw.find('day;'):]), sep=';')
mdf = pd.read_csv(io.StringIO(
    'timestamp;product;fair_price\n' +
    '\n'.join(
        m.group(1) for m in re.compile('<csv:([^>]*)>').finditer(raw)
    )
), sep=';')

# hdf = pd.read_csv('history.csv', sep=';')
# mdf = pd.read_csv('mylog.csv', sep=';')

no_suffix = ['day', 'timestamp', 'product']
products = ['PEARLS', 'BANANAS', 'COCONUTS', 'PINA_COLADAS']

df = pd.DataFrame(list(range(0, 100000, 100)), columns=['timestamp'])
for p in products:
    df = df.merge(
        hdf[hdf['product'] == p].rename(columns={c: c + '_' + p for c in hdf.columns if c not in no_suffix}).drop(columns=['day', 'product']),
        how='inner', on='timestamp'
    )
    df = df.merge(
        mdf[mdf['product'] == p].rename(columns={c: c + '_' + p for c in mdf.columns if c not in no_suffix}).drop(columns=['product']),
        how='inner', on='timestamp'
    )

print(df.columns)
print(df.head())

df.plot(x='timestamp', y=['profit_and_loss_PEARLS', 'profit_and_loss_BANANAS', 'profit_and_loss_COCONUTS', 'profit_and_loss_PINA_COLADAS'])

df.plot(x='timestamp', y=['mid_price_BANANAS', 'bid_price_1_BANANAS', 'ask_price_1_BANANAS', 'fair_price_BANANAS'])
plt.xlim(0, 10000)

plt.show()

# price = price * base ^ -(max(x - Âµ - b, 0))
