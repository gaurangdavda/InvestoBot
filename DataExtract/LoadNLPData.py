import pandas as pd
import numpy as np
import os
import csv
import json

class LoadNLPData():

    def __init__(self):
        self.nlpdata = self.loadJson()

    def loadCsv(self):
        self.nlpdata = pd.read_csv(os.getcwd()+'/Data/NLPData.csv')

    def loadJson(self):
        with open(os.getcwd()+'/Data/convertcsv.json') as file:
            return json.load(file)

    def convertCsvToJson(self):
        self.loadCsv()
        self.updatedNlpData = self.nlpdata[['TAGS','PATTERNS']]

    def getResponsesFromTag(self, tag):
        for tg in self.nlpdata["intents"]:
            if tg['tag'] == tag:
                responses = tg['response']
                return responses