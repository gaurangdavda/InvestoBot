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

@app.route('/chat', methods=['POST'])
def chat():
    if not request.json or extractMessageAndUserFromRequest(request)[0].lower() == 'quit':
        return jsonify('Thank you!'), 400
    if extractMessageAndUserFromRequest(request)[0].isnumeric():
        handleAccountNumberOnly(extractMessageAndUserFromRequest(request)[0])
    message, userId = extractMessageAndUserFromRequest(request)
    bag = performPreProcessingAndTransformations(message)
    predictedTag = predictTag(bag)
    prevTag, prevQues = updateTagAndQuestionDict(predictedTag, userId, message)
    print(predictedTag)
    responses = loadNLPDataObject.getResponsesFromTag(predictedTag)
    if responses:
        if len(responses) == 0:
            handleYesNoTags()
        elif len(responses) == 1:
            handleSingleResponse()
        elif len(responses) == 2:
            handleMultipleResponses(responses, predictedTag, message)
    return jsonify(predictedTag), 400

def extractMessageAndUserFromRequest(request):
    return request.json['message'],request.json['userId']

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
    if results[results_index] < 0.5:
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

def handleYesNoTags():
    pass

def handleSingleResponse():
    pass

def handleMultipleResponses(responses, tag, message):
    if tag == 'accountNumberEntered':
        accountNumber = 0
        ws = message.split(' ')
        for w in ws:
            if w.isnumeric():
                accountNumber = w
                break
        return handleAccountNumberOnly(accountNumber)
    return 'I am sorry. I did not understand. Can you try by altering the sentence?'

def handleAccountNumberOnly(accountNumber):
    accountDetails = loadPortfolioDataObject.getAccountDetails(accountNumber)
    if not accountDetails:
        return 'Unfortunately we cannot find your account details in our database. Kindly re-enter the account number.'
    return handleValidAccountNumber(accountDetails)

def handleValidAccountNumber(accountDetails):
    optimizePortfolioObject.loadCurrentPortfolio(accountDetails)
    return "Thank you. This is how your portfolio looks currently. Would you like to optimize it for higher gains?"

if __name__ == "__main__": 
	app.run(host='0.0.0.0',port=5000,debug=True)