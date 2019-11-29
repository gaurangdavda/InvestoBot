import re
import numpy as np
import pandas as pd
import os
from alpha_vantage.timeseries import TimeSeries

class OptimizePortfolio():

    def __init__(self):
        self.ts = TimeSeries(key, output_format='pandas')

    def loadCurrentPortfolio(self, accountDetails):
        pass

    def getLiveStockRate(self, stockCode):
        df = self.ts.get_intraday(stockCode, interval='1min', outputsize='full')