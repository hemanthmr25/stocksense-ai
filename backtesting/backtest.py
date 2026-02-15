import backtrader as bt

class MLStrategy(bt.Strategy):
    def next(self):
        if not self.position:
            if self.datas[0].close[0] > self.datas[0].close[-1]:
                self.buy()
        else:
            if self.datas[0].close[0] < self.datas[0].close[-1]:
                self.sell()
