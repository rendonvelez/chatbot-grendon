#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import random
import sys
import json
import requests
import numpy as np
import tensorflow as tf

from flask import Flask, request

data_path = "model/human_text.txt"
data_path2 = "model/robot_text.txt"
# Defining lines as a list of each line
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
with open(data_path2, 'r', encoding='utf-8') as f:
    lines2 = f.read().split('\n')
lines = [re.sub(r"\[\w+\]", 'hola', line.lower()) for line in lines]
lines = [" ".join(re.findall(r"\w+", line.lower())) for line in lines]
lines2 = [re.sub(r"\[\w+\]", '', line.lower()) for line in lines2]
lines2 = [" ".join(re.findall(r"\w+", line.lower())) for line in lines2]
# grouping lines by response pair
pairs = list(zip(lines, lines2))
# random.shuffle(pairs)

input_docs = []
target_docs = []
input_tokens = set()
target_tokens = set()
for line in pairs:
    input_doc, target_doc = line[0], line[1]
    # Appending each input sentence to input_docs
    input_docs.append(input_doc)
    # Splitting words from punctuation
    target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
    # Redefine target_doc below and append it to target_docs
    target_doc = '<START> ' + target_doc + ' <END>'
    target_docs.append(target_doc)

    # Now we split up each sentence into words and add each unique word to our vocabulary set
    for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
        if token not in input_tokens:
            input_tokens.add(token)
    for token in target_doc.split():
        if token not in target_tokens:
            target_tokens.add(token)

input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

input_features_dict = dict(
    [(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict(
    [(token, i) for i, token in enumerate(target_tokens)])
reverse_input_features_dict = dict(
    (i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict(
    (i, token) for token, i in target_features_dict.items())

training_model = tf.keras.models.load_model('model/training_model.h5')

app = Flask(__name__)


@app.route('/', methods=['GET'])
def verify():

    # cuando el endpoint este registrado como webhook, debe mandar de vuelta
    # el valor de 'hub.challenge' que recibe en los argumentos de la llamada

    if request.args.get('hub.mode') == 'subscribe' \
            and request.args.get('hub.challenge'):
        if not request.args.get('hub.verify_token') \
                == os.environ['VERIFY_TOKEN']:
            return ('Verification token mismatch', 403)
        return (request.args['hub.challenge'], 200)

    return ('Hello world', 200)


@app.route('/', methods=['POST'])
def webhook():

    # endpoint para procesar los mensajes que llegan

    data = request.get_json()

    # log(data)  # logging, no necesario en produccion

    inteligente = True

    if data['object'] == 'page':

        for entry in data['entry']:
            for messaging_event in entry['messaging']:

                if messaging_event.get('message'):  # alguien envia un mensaje

                    # el facebook ID de la persona enviando el mensaje
                    sender_id = messaging_event['sender']['id']
                    # el facebook ID de la pagina que recibe (tu pagina)
                    recipient_id = messaging_event['recipient']['id']
                    # el texto del mensaje
                    message_text = messaging_event['message']['text']

                    if inteligente:
                        send_message(sender_id, 'Tonces')
                    else:
                        send_message(sender_id, 'Hola')

                if messaging_event.get('delivery'):  # confirmacion de delivery
                    pass

                if messaging_event.get('optin'):  # confirmacion de optin
                    pass

                # evento cuando usuario hace click en botones
                if messaging_event.get('postback'):
                    pass

    return ('ok', 200)


def send_message(recipient_id, message_text):

    # log("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))

    params = {'access_token': os.environ['PAGE_ACCESS_TOKEN']}
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({'recipient': {'id': recipient_id},
                       'message': {'text': message_text}})

    r = requests.post('https://graph.facebook.com/v2.6/me/messages',
                      params=params, headers=headers, data=data)
    if r.status_code != 200:
        log(r.status_code)
        log(r.text)


def log(message):  # funcion de logging para heroku
    print(str(message))
    sys.stdout.flush()


if __name__ == '__main__':
    app.run(debug=True)
