from flask import Flask, jsonify, request, abort
import requests

VERIFICATION_TOKEN = "9ccxxxxxxxxxxxxxxxxxxxxxxxx"
ACCESS_TOKEN = "xoxb-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcome to Investobot" , 200

@app.route('/', methods=['POST'])
def create_task():
    if not request.json or not 'type' in request.json:
        print(jsonify(request.json))
        abort(400)
    elif request.json['type'] == "url_verification":
        token = { 'challenge': request.json['challenge'] }
        return jsonify(token), 200
    elif request.json['type'] == "event_callback":
        print(request.json)
        message = {
            'token' : ACCESS_TOKEN,
            'channel': request.json['event']['channel'],
            'text': request.json['event']['blocks'][0]['elements'][0]['elements'][1]['text']
        }

        response = requests.get(
             'https://slack.com/api/chat.postMessage',
              params = message,
              headers={'Content-type': 'application/json'},
        )
        print(response)


        return jsonify(message), 200



if __name__ == "__main__":
    app.run(host="0.0.0.0",port=80)
