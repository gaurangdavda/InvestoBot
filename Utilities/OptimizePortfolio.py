import re
import numpy as np
import pandas as pd
import os
from alpha_vantage.timeseries import TimeSeries
from collections import OrderedDict
from datetime import datetime
from scipy.optimize import minimize

class OptimizePortfolio():

    def __init__(self):
        self.ts = TimeSeries('P9HHDBDFZ3RZYTSW', output_format='pandas')
        self.stockDict = {'GOOGL':'Google', 'MSFT':'Microsoft','AAPL':'Apple','FB':'Facebook','TWTR':'Twitter'}

    def loadCurrentPortfolio(self, accountDetails):
        stocks = ['GOOGL','MSFT','TWTR','FB','AAPL']
        # currentEvaluation = 0
        # for s in stocks:
        #     price = self.getLiveStockRate(s)
        #     alloc = accountDetails[s][0]
        #     currentEvaluation += alloc * 10 * price
        currentEvaluation = "Thank you! The current portfolio allocations are as follows:"
        for s in stocks:
            if accountDetails[s][0] > 0.0:
                currentEvaluation += " " + str(accountDetails[s][0] * 100) + "% stocks of " + self.stockDict.get(s) + ","
        currentEvaluation = currentEvaluation.strip(',')
        return currentEvaluation

    def getLiveStockRate(self, stockCode):
        df = self.ts.get_intraday(stockCode, interval='1min', outputsize='compact')
        df = df[0]['4. close']
        return df[len(df)-1]

    def getDailyStockRate(self, stockCode):
        # df,_ = self.ts.get_daily(stockCode, outputsize='full')
        df = pd.read_csv(os.getcwd() + "/Data/daily_"+stockCode+".csv", index_col='date')
        df = df.rename(columns={'1. open': 'Open', '2. high':'High', '3. low':'Low', '4. close':'Close', '5. volume':'Volume'})
        return df

    def optimizePortfolio(self, accountNumber, loadPortfolioDataObject):
        print(accountNumber)
        accountDetails = loadPortfolioDataObject.getAccountDetails(accountNumber)
        stocks = ['GOOGL','MSFT','TWTR','FB','AAPL']
        stockAllocations = []
        dfList = []
        for s in stocks:
            dfList.append(self.getDailyStockRate(s))
            stockAllocations.append((s, accountDetails[s][0]))
        
        companyAllocations = OrderedDict(stockAllocations)
        startDate = '2019-01-01'
        endDate = datetime.today().strftime('%Y-%m-%d')
        dates = pd.date_range(startDate, endDate)
        newdf = pd.DataFrame(index = dates)
        for id, df in enumerate(dfList):
            df.drop(['Open', 'High', 'Low', 'Volume'], axis=1, inplace=True)
            df = df.rename(columns={'Close':stocks[id]})
            newdf = newdf.join(df)
        
        newdf = newdf.dropna()
        afterdf = newdf
        norm = newdf/newdf.iloc[0]
        norm = norm.astype(float)
        for s in stocks:
            norm[s] = norm[s].astype(float)
            norm[s] *= float(companyAllocations.get(s)*accountDetails['TOTAL'][0])
        
        evaluation = norm.sum(axis=1)
        before, beforeSharpeRatio, beforePortValue = self.printStats(evaluation)

        baseAllocation = []
        for _, value in companyAllocations.items():
            baseAllocation.append(value)
        baseAllocation = np.array(baseAllocation)
        self.currentValue = accountDetails['TOTAL'][0]
        alloc = minimize(self.calculateSharpeRatio, baseAllocation, args=(afterdf,),method='SLSQP',
        bounds=((0,1),(0,1),(0,1),(0,1),(0,1)),constraints=({'type':'eq','fun':lambda inputs: 1.0-np.sum(inputs)}))

        dates = pd.date_range(startDate, endDate)
        newdf = pd.DataFrame(index = dates)
        for id, df in enumerate(dfList):
            df = df.rename(columns={'Close':stocks[id]})
            newdf = newdf.join(df)
        newdf = newdf.dropna()
        norm = newdf/newdf.iloc[0]
        norm = norm.astype(float)
        for id, s in enumerate(stocks):
            norm[s] = norm[s].astype(float)
            norm[s] *= float(alloc.x[id]*accountDetails['TOTAL'][0])
        evaluationNew = norm.sum(axis=1)
        after, afterSharpeRatio, afterPortValue = self.printStats(evaluationNew)

        s1 = "The current portfolio allocations are as follows:"
        for s in stocks:
            if accountDetails[s][0] > 0.0:
                s1 += " " + str(accountDetails[s][0] * 100) + "% stocks of " + self.stockDict.get(s) + ","
        s1 = s1.strip(',')
        s1 += ". Based on these allocations, the current portfolio returns are as follows. " + before
        s1 +=  ". However, on optimizing it, the new portfolio returns are as follows. " + after
        s1 += ". These new returns can be achieved if portfolio allocations change to the following:"
        for id,s in enumerate(stocks):
            if alloc.x[id] > 0.0:
                s1 += " " + str(alloc.x[id] * 100) + "% stocks of " + self.stockDict.get(s) + ","
        s1 = s1.strip(',')
        if afterPortValue > beforePortValue:
            s1 += ". The optimized portfolio is $" + str(afterPortValue - beforePortValue) + " higher than original portfolio."
        else:
            s1 += ". The optimized portfolio is $" + str(beforePortValue - afterPortValue) + " less than original portfolio as original portfolio had higher risks and the optimization process tries to balance the risk involved."
        s1 += " Kindly drop by our office, if you want to update your portfolio allocations. Kindly type quit to exit or type Hello to continue chatting with me."
        return s1

    def printStats(self, evaluation):
        cumulativeReturn = (evaluation[-1]/evaluation[0])-1

        dailyReturn = evaluation.copy()
        dailyReturn[1:] = (evaluation[1:]/evaluation[:-1].values)-1
        dailyReturn = dailyReturn[1:]

        dailyReturnAverage = dailyReturn.mean()

        dailyReturnStd = dailyReturn.std()

        sharpeRatio = dailyReturnAverage/dailyReturnStd
        sharpeRatio = np.sqrt(len(evaluation))*sharpeRatio

        portValue = evaluation.iloc[len(evaluation)-1]
        
        s1 = "Cumulative Return is $" + str(cumulativeReturn)
        s1 = s1 + ". Daily Return mean is $" + str(dailyReturnAverage)
        s1 = s1 + ". Volatility or Standard Deviation is " + str(dailyReturnStd)
        s1 = s1 + ". Sharpe Ratio is " + str(sharpeRatio)
        s1 = s1 + ". Portfolio evaluation at the end of the term is $" + str(portValue) + "."
        
        return s1, sharpeRatio, portValue

    def calculateSharpeRatio(self, allocs, afterdf):
        norm = afterdf/afterdf.iloc[0]
        allocated = norm * allocs
        newValue = allocated * self.currentValue
        evaluationNew = newValue.sum(axis=1)
        
        dailyReturn = evaluationNew.copy()
        dailyReturn[1:] = (evaluationNew[1:]/evaluationNew[:-1].values)-1
        dailyReturn = dailyReturn[1:]
        sharpeRatio = dailyReturn.mean()/dailyReturn.std()
        sharpeRatio = np.sqrt(len(evaluationNew))*sharpeRatio
        return -1*sharpeRatio