import pandas as pd
import numpy as np
import os
import csv

class LoadNLPData():

    def __init__(self):
        pass

    def loadCsv(self):
        self.nlpdata = pd.read_csv(os.getcwd()+'/Data/NLPData.csv')

    def convertCsvToJson(self):
        self.loadCsv()
        self.updatedNlpData = self.nlpdata[['TAGS','PATTERNS']]

LoadNLPData().convertCsvToJson()