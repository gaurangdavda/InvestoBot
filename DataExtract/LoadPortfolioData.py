import pandas as pd
import numpy as np
import os
import csv
import json

class LoadPortfolioData():

    def __init__(self):
        self.portfoliodata = self.loadCsv()

    def loadCsv(self):
        df = pd.read_csv(os.getcwd()+'/Data/portfolio_data.csv')
        df = df.fillna(0)
        return df

    def getAccountDetails(self, accountNumber):
        df = self.portfoliodata[self.portfoliodata['ACCOUNT_NO'] == accountNumber]
        if len(df) == 0:
            return pd.DataFrame()
        else:
            df.reset_index(drop=True, inplace=True)
        return df

    def isValidAccount(self, accountNumber):
        if accountNumber in self.portfoliodata['ACCOUNT_NO']:
            print(self.portfoliodata['ACCOUNT_NO'])
            return True
        return False