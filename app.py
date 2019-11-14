import pandas as pd
import numpy as np
import os
import json
import csv
import requests
import logging
import re
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def getMessage():
    print('Start chatting with IB. Enter (quit) to close the chatbot.')
    while True:
        inp = input("You: ")
        if inp.lower() == 'quit':
            break
        if inp.lower() == 'hello':
            print('Hi!!')
    return jsonify('Thank you.')

if __name__ == "__main__":
	app.run(host='0.0.0.0',port=5000,debug=True)