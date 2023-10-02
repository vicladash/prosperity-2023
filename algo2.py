from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, Listing, Trade
import numpy as np
import pandas as pd
from math import copysign


Time = int
Symbol = str
Product = str
Position = int
UserId = str
Observation = int


class ProductTrader:
    def __init__(self, symbol, passive_curve, active_curve, book_skew_factor, retreat_rate, retreat_bias, fair_decay_rate, position_limit, risk_limit):
        # Configuration
        self.symbol = symbol
    
        self.passive_curve = passive_curve
        self.active_curve = active_curve
    
        self.book_skew_factor = book_skew_factor
        self.retreat_rate = retreat_rate
        self.retreat_bias = retreat_bias
        self.fair_decay_rate = fair_decay_rate

        self.position_limit = position_limit
        self.risk_limit = risk_limit

        # Transactional
        self.balance = 0
        self.fair_price = None

        # History
        self.fair_price_history = []
        self.best_bid_history = []
        self.best_ask_history = []

        # Debug
        self.debug_orders = True
    
    def log(self, book, position, own_trades: List[Trade], timestamp):
        print('TRADES:', own_trades)
        print(f'{self.symbol} BALANCE: {self.balance} POSITION: {position} FAIR PRICE: {self.fair_price}')
        print(f'<csv:{timestamp};{self.symbol};{self.fair_price}>')
    
    def final_log(self):
        pass

    def update(self, book, position, own_trades: List[Trade]):
        buy_trades = [t for t in own_trades if t.buyer == 'SUBMISSION']
        sell_trades = [t for t in own_trades if t.seller == 'SUBMISSION']

        best_bid = max(book.buy_orders[:,0])
        best_ask = min(book.sell_orders[:,0])
        mid_price = (best_bid + best_ask) / 2

        self.balance += (
            sum([t.price * t.quantity for t in sell_trades]) -
            sum([t.price * t.quantity for t in buy_trades])
        )

        fair_price = mid_price
        fair_price += (
            # positive quantity * positive weight
            sum(-book.sell_orders[:,1] * np.exp(-1 * (book.sell_orders[:,0] - mid_price))) -
            sum(book.buy_orders[:,1] * np.exp(-1 * (mid_price - book.buy_orders[:,0])))
        ) * self.book_skew_factor
        fair_price *= (1 + self.retreat_rate) ** (
            # sum([t.quantity for t in sell_trades]) -
            # sum([t.quantity for t in buy_trades])
            -copysign(max(abs(position) - self.retreat_bias, 0), position)
        )
        if self.fair_price is None:
            self.fair_price = fair_price
        else:
            self.fair_price = self.fair_price * (1 - self.fair_decay_rate) + fair_price * self.fair_decay_rate
        
        self.fair_price_history.append(self.fair_price)
        self.best_bid_history.append(best_bid)
        self.best_ask_history.append(best_ask)

    def passive_trade(self, book, position, curr_orders: List[Order]) -> List[Order]:
        result = []
        best_bid = max(book.buy_orders[:,0])
        best_ask = max(book.sell_orders[:,0])

        curr_bid_quantity = sum([o.quantity for o in curr_orders if o.quantity > 0])
        remaining_bids = self.position_limit - position - curr_bid_quantity
        if position + curr_bid_quantity >= self.risk_limit:
            remaining_bids = 0
        remaining_bids = max(remaining_bids, 0)

        for price_delta, quantity in self.passive_curve:
            bid_price = self.fair_price - price_delta
            # Ensure bid is passive, doesn't cross the book
            if bid_price >= best_ask:
                continue
            bid_quantity = min(quantity, remaining_bids)

            assert bid_quantity >= 0
            if bid_quantity > 0:
                result.append(Order(self.symbol, bid_price, bid_quantity))
                remaining_bids -= bid_quantity
                if self.debug_orders: print('Passive:', Order(self.symbol, bid_price, bid_quantity))
        
        curr_ask_quantity = sum([-o.quantity for o in curr_orders if o.quantity < 0])
        remaining_asks = self.position_limit - (-position) - curr_ask_quantity
        if position - curr_ask_quantity <= -self.risk_limit:
            remaining_asks = 0
        remaining_asks = max(remaining_asks, 0)
        
        for price_delta, quantity in self.passive_curve:
            ask_price = self.fair_price + price_delta
            # Ensure ask is passive, doesn't cross the book
            if ask_price <= best_bid:
                continue
            ask_quantity = min(quantity, remaining_asks)

            assert ask_quantity >= 0
            if ask_quantity > 0:
                result.append(Order(self.symbol, ask_price, -ask_quantity))
                remaining_asks -= ask_quantity
                if self.debug_orders: print('Passive:', Order(self.symbol, ask_price, -ask_quantity))
        
        return result

    def active_trade(self, book, position, curr_orders: List[Order]) -> List[Order]:
        result = []
        best_bid = max(book.buy_orders[:,0])
        best_ask = max(book.sell_orders[:,0])

        remaining_bids = min(
            self.position_limit - position - sum([o.quantity for o in curr_orders if o.quantity > 0]),
            self.risk_limit - position
        )
        remaining_bids = max(remaining_bids, 0)

        for price_delta, quantity in self.active_curve[::-1]:
            bid_price = self.fair_price - price_delta
            # Ensure bid is active, crosses the book
            if bid_price < best_ask:
                continue
            bid_quantity = min(quantity, remaining_bids)

            assert bid_quantity >= 0
            if bid_quantity > 0:
                result.append(Order(self.symbol, bid_price, bid_quantity))
                remaining_bids -= bid_quantity
                if self.debug_orders: print('Active:', Order(self.symbol, bid_price, bid_quantity))
        
        remaining_asks = min(
            self.position_limit - (-position) - sum([-o.quantity for o in curr_orders if o.quantity < 0]),
            position - (-self.risk_limit)
        )
        remaining_asks = max(remaining_asks, 0)

        for price_delta, quantity in self.active_curve:
            ask_price = self.fair_price + price_delta
            # Ensure ask is active, crosses the book
            if ask_price > best_bid:
                continue
            ask_quantity = min(quantity, remaining_asks)

            assert ask_quantity >= 0
            if ask_quantity > 0:
                result.append(Order(self.symbol, ask_price, -ask_quantity))
                remaining_asks -= ask_quantity
                if self.debug_orders: print('Active:', Order(self.symbol, ask_price, -ask_quantity))
        
        return result

    def run(self, state: TradingState) -> List[Order]:
        book: OrderDepth = state.order_depths[self.symbol]
        book.buy_orders = np.array(list(book.buy_orders.items()))
        book.sell_orders = np.array(list(book.sell_orders.items()))

        own_trades: List[Trade] = state.own_trades.get(self.symbol, [])
        own_trades = [t for t in own_trades if t.timestamp == state.timestamp - 100]

        position: Position = state.position.get(state.listings[self.symbol]['product'], 0)

        print('----------------------------------------', state.timestamp)
        self.update(book, position, own_trades)
        self.log(book, position, own_trades, state.timestamp)

        if state.timestamp >= 500:
            orders = self.active_trade(book, position, [])
            orders += self.passive_trade(book, position, orders)

            return orders
        else:
            return []


class Trader:
    def __init__(self):
        self.pearlsTrader = ProductTrader(
            symbol='PEARLS',
            passive_curve=[
                (2, 2),
                (3, 6),
                (4, 10)
            ],
            active_curve=[
                (1, 2),
                (2, 5),
                (3, 6),
            ],
            book_skew_factor=0,
            retreat_rate=5e-6,
            retreat_bias=0,
            fair_decay_rate=0.05,
            position_limit=20,
            risk_limit=20
        )
        self.bananasTrader = ProductTrader(
            symbol='BANANAS',
            passive_curve=[
                (3, 3),
                (4, 4),
                (5, 5)
            ],
            active_curve=[
                (2, 2)
            ],
            book_skew_factor=0.2,
            retreat_rate=2e-5,
            retreat_bias=20,
            fair_decay_rate=0.15,
            position_limit=20,
            risk_limit=20
        )
        self.coconutsTrader = ProductTrader(
            symbol='COCONUTS',
            passive_curve=[
                (3, 3),
                (4, 4),
                (5, 5)
            ],
            active_curve=[
                (2, 2)
            ],
            book_skew_factor=0.2,
            retreat_rate=0.00002,
            retreat_bias=5,
            fair_decay_rate=0.09,
            position_limit=600,
            risk_limit=20
        )
        self.pinaColadasTrader = ProductTrader(
            symbol='PINA_COLADAS',
            passive_curve=[
                (3, 3),
                (4, 4),
                (5, 5)
            ],
            active_curve=[
                (2, 2)
            ],
            book_skew_factor=0.2,
            retreat_rate=0.00002,
            retreat_bias=5,
            fair_decay_rate=0.1,
            position_limit=300,
            risk_limit=20
        )

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {
            'PEARLS': self.pearlsTrader.run(state),
            'BANANAS': self.bananasTrader.run(state),
            'COCONUTS': self.coconutsTrader.run(state),
            'PINA_COLADAS': self.pinaColadasTrader.run(state)
        }

        if state.timestamp == 99900:
            print('\n\n')
            self.pearlsTrader.final_log()
            self.bananasTrader.final_log()
            self.coconutsTrader.final_log()
            self.pinaColadasTrader.final_log()

        return result


if __name__ == '__main__':
	t = Trader()
	print(t.run(TradingState(
		timestamp=0,
		# listings={'TEST': Listing('TEST', 'TEST', 'TestProduct')},\
		listings={'PEARLS': {'symbol': 'PEARLS', 'product': 'PEARLS', 'denomination': 'PEARLSProduct'}},
		order_depths={'PEARLS': OrderDepth(buy_orders={108:15,109:13,110:11}, sell_orders={112:11,113:13,114:15})},
		own_trades={'PEARLS': []},
		market_trades={'PEARLS': []},
		position={'PEARLS': -5},
		observations={},
	)))
