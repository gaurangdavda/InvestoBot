import pandas as pd
import numpy as np
import os
import json
import csv
import pickle
import requests
import logging
import re
from flask import Flask, jsonify, request, abort
from Utilities.PreProcessMessage import PreProcessMessage
from Utilities.OptimizePortfolio import OptimizePortfolio
from DataExtract.LoadNLPData import LoadNLPData
from DataExtract.LoadPortfolioData import LoadPortfolioData
import tensorflow as tf
import tflearn
import nltk
from nltk.stem.lancaster import LancasterStemmer


app = Flask(__name__)

# loading pickled model and data
with open(os.getcwd()+"/models/data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.load(os.getcwd()+"/models/model.tflearn")

preProcessObject = PreProcessMessage()
loadNLPDataObject = LoadNLPData()
loadPortfolioDataObject = LoadPortfolioData()
optimizePortfolioObject = OptimizePortfolio()
tagDict = {}
quesDict = {}
stockDict = {'GOOGL':'Google', 'MSFT':'Microsoft','AAPL':'Apple','FB':'Facebook','TWTR':'Twitter'}
VERIFICATION_TOKEN = "9ccrsm3vdhMnEvDBNNmqodRC"
ACCESS_TOKEN = "xoxb-825935562227-839955072534-TQ1vQmnGxhQ3mscKcyfIeLOf"

@app.route('/', methods=['POST'])
def chat():
    if not request.json or not 'type' in request.json:
        abort(400)
    elif request.json['type'] == "url_verification":
        token = { 'challenge': request.json['challenge'] }
        return jsonify(token), 200
    elif request.json['type'] == "event_callback":
        message, userId = extractMessageAndUserFromRequest(request)
        if message.lower() == 'quit':
            postResponseToSlack('Thank you!')
            return jsonify('Ok'), 200
        if message.isnumeric():
            resp = handleAccountNumberOnly(int(message))
            postResponseToSlack(resp)
            return jsonify('Ok'), 200
        bag = performPreProcessingAndTransformations(message)
        predictedTag = predictTag(bag)
        prevTag, prevQues = updateTagAndQuestionDict(predictedTag, userId, message)
        print(predictedTag)
        responses = loadNLPDataObject.getResponsesFromTag(predictedTag)
        if responses:
            if len(responses) == 0:
                predictedTag = handleYesNoTags(predictedTag)
            elif len(responses) == 1:
                predictedTag = handleSingleResponse(responses, predictedTag, message, prevTag, prevQues, userId)
            elif len(responses) == 2:
                predictedTag = handleMultipleResponses(responses, predictedTag, message)
        postResponseToSlack(predictedTag)
        return jsonify(predictedTag), 200

def postResponseToSlack(response):
    message = {
            'token' : ACCESS_TOKEN, 'channel': request.json['event']['channel'], 'text': response
        }
    response = requests.get('https://slack.com/api/chat.postMessage', params = message, 
        headers={'Content-type': 'application/json'}
        )

def extractMessageAndUserFromRequest(request):
    message = request.json['event']['blocks'][0]['elements'][0]['elements'][1]['text']
    userId = request.json['event']['blocks'][0]['elements'][0]['elements'][0]['user_id']
    return message.strip(), userId

def performPreProcessingAndTransformations(message):
    cleanedMessage = preProcessObject.preprocess_message(message)
    bag = preProcessObject.bagOfWords(cleanedMessage, words)
    return bag

def predictTag(bag):
    tag = ''
    results = model.predict([bag])[0]
    results_index = np.argmax(results)
    print(results[results_index]) 
    print(labels[results_index])
    if results[results_index] < 0.6:
        return 'I am sorry. I did not understand. Can you try by altering the sentence?'
    tag = labels[results_index]
    return tag

def updateTagAndQuestionDict(newTag, userId, message):
    if userId in tagDict.keys():
        prevTag = tagDict.get(userId)
        prevQues = quesDict.get(userId)
        tagDict[userId] = newTag
        quesDict[userId] = message
        return prevTag, prevQues
    else:
        tagDict[userId] = newTag
        quesDict[userId] = message
        return None, None

def handleYesNoTags(predictedTag):
    return predictedTag

def handleSingleResponse(responses, predictedTag, message, prevTag, prevQues, userId):
    if predictedTag == 'yesPortfolioOptimize':
        accountNumber = extractNumberFromMessage(prevQues)
        predictedTag = optimizePortfolioObject.optimizePortfolio(accountNumber, loadPortfolioDataObject)
    elif predictedTag == 'name':
        ws = message.lower().split()
        name = ''
        for w in ws:
            if not (w == 'is' or w == 'my' or w == 'name' or w == 'full' or w == 'first' or w == 'last' or w == 'middle' or w == 'i' or w == 'am'):
                name += w.capitalize() + " "
        name = name.strip()
        predictedTag = "Thank you "+ name +"! Do you have an investment portfolio account with us?"
    elif predictedTag == 'yesStockInvestment':
        ws = message.lower().split()
        stockList = []
        codeList = list(map(lambda x:x.lower(),list(stockDict.keys())))
        nameList = list(map(lambda x:x.lower(),list(stockDict.values())))
        for w in ws:
            if w in codeList:
                stockList.append(w.upper())
            elif w in nameList:
                stockList.append([k for k,v in stockDict.items() if v == w.capitalize()][0])
        if len(stockList) == 0:
            stockList = list(stockDict.keys())
        priceList = []
        predictedTag = "Awesome! Below is some information you maybe interested in. "
        for s in stockList:
            price = optimizePortfolioObject.getLiveStockRate(s)
            priceList.append(price)
            predictedTag += "The live rate for " + stockDict.get(s) + "'s stock is $" + str(price) + ". "
        predictedTag = predictedTag.strip()
        predictedTag += " I hope this helps you and you create a portfolio with us. If you need to create a portfolio or know more about stock investment, kindly contact our office. Kindly type quit to exit or type Hello to continue chatting with me."
        return predictedTag
    else:
        predictedTag = responses[0]
    return predictedTag

def extractNumberFromMessage(message):
    number = 0
    ws = message.split(' ')
    for w in ws:
        if w.isnumeric():
            number = int(w)
            break
    return number

def handleMultipleResponses(responses, tag, message):
    if tag == 'accountNumberEntered':
        accountNumber = extractNumberFromMessage(message)
        return handleAccountNumberOnly(int(accountNumber))
    return 'I am sorry. I did not understand. Can you try by altering the sentence?'

def handleAccountNumberOnly(accountNumber):
    accountDetails = loadPortfolioDataObject.getAccountDetails(accountNumber)
    if len(accountDetails) == 0:
        return 'Unfortunately we cannot find your account details in our database. Kindly re-enter the account number.'
    return handleValidAccountNumber(accountDetails)

def handleValidAccountNumber(accountDetails):
    currentEvaluation = optimizePortfolioObject.loadCurrentPortfolio(accountDetails)
    s1 = currentEvaluation + ". Would you like to optimize it for higher gains?"
    return s1

if __name__ == "__main__": 
	app.run(host='0.0.0.0',port=5000,debug=True)