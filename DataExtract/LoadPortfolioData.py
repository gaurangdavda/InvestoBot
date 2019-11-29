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
        if self.isValidAccount(accountNumber):
            for i,r in self.portfoliodata.iterrows():
                if r['ACCOUNT_NO'] == accountNumber:
                    return r
        return None

    def isValidAccount(self, accountNumber):
        if accountNumber in self.portfoliodata['ACCOUNT_NO']:
            return True
        return False