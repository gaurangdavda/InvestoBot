import pandas as pd
import numpy as np
import os
import csv

class LoadNLPData():

    def __init__(self):
        pass

    def loadCsv(self):
        self.nlpdata = pd.read_csv('../Data/NLPdata.csv')

    def convertCsvToJson(self):
        self.loadCsv()