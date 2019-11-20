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


app = Flask(__name__)

# loading pickled models
# nlpModel = pickle.loads('')

tagDict = {}
quesDict = {}

@app.route('/chat', methods=['POST'])
def chat():
    if not request.json or extractMessageAndUserFromRequest(request)[0].lower() == 'quit':
        return jsonify('Thank you!'), 400
    message, userId = extractMessageAndUserFromRequest(request)
    transformedMessage = performPreProcessingAndTransformations(message)
    predictedTag = predictTag(transformedMessage)
    prevTag, prevQues = updateTagAndQuestionDict(predictedTag, userId, message)
    return jsonify(request.json)

def extractMessageAndUserFromRequest(request):
    return request.json['message'],request.json['userId']

def performPreProcessingAndTransformations(message):
    cleanedMessage = PreProcessMessage().preprocess_message(message)
    transformedMessage = PreProcessMessage().transformCleanedMessage(cleanedMessage)
    return transformedMessage

def predictTag(transformedMessage):
    tag = ''
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

if __name__ == "__main__": 
	app.run(host='0.0.0.0',port=5000,debug=True)