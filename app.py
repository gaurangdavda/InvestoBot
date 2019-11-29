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
            handleAccountNumberOnly(message)
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
    return message, userId

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
    if results[results_index] < 0.7:
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