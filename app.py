#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import random
import sys
import json
import requests
import numpy as np
from keras.layers import Input, LSTM, Dense
from keras.models import load_model, Model

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

# Maximum length of sentences in input and target documents
max_encoder_seq_length = max(
    [len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max(
    [len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])

# Dimensionality
dimensionality = 256

# Encoder
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(dimensionality, return_state=True)
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell]
# Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(
    decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

training_model = load_model('model/training_model.h5')
encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

latent_dim = 256
decoder_state_input_hidden = Input(shape=(latent_dim,))
decoder_state_input_cell = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
decoder_outputs, state_hidden, state_cell = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_hidden, state_cell]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs] + decoder_states)

negative_responses = ("no", "negativo", "nada", "nop", "nopi")
exit_commands = ("chao", "adios", "nos vemos",
                 "suerte", "chaito", "bye", "chaolin")

sessionid = 0

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
                    message_text = messaging_event['message']['text'].lower()

                    if inteligente:
                        if make_exit(message_text):
                            send_message(
                                sender_id, "Que tengas un hermoso día!")
                        elif message_text in negative_responses:
                            send_message(
                                sender_id, "Ok, perfecto!")
                        else:
                            send_message(
                                sender_id, generate_response(message_text))
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


def decode_response(test_input):
    # Getting the output states to pass into the decoder
    log("Pruebas")
    states_value = encoder_model.predict(test_input)
    # Generating empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Setting the first token of target sequence with the start token
    target_seq[0, 0, target_features_dict['<START>']] = 1.

    # A variable to store our response word by word
    decoded_sentence = ''

    stop_condition = False
    # while not stop_condition:
    #     # Predicting output tokens with probabilities and states
    #     output_tokens, hidden_state, cell_state = decoder_model.predict(
    #         [target_seq] + states_value)
    # # Choosing the one with highest probability
    #     sampled_token_index = np.argmax(output_tokens[0, -1, :])
    #     sampled_token = reverse_target_features_dict[sampled_token_index]
    #     decoded_sentence += " " + sampled_token
    # # Stop if hit max length or found the stop token
    #     if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
    #         stop_condition = True
    # # Update the target sequence
    #     target_seq = np.zeros((1, 1, num_decoder_tokens))
    #     target_seq[0, 0, sampled_token_index] = 1.
    #     # Update states
    #     states_value = [hidden_state, cell_state]
    return decoded_sentence

# Method to convert user input into a matrix


def string_to_matrix(user_input):
    tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
    user_input_matrix = np.zeros(
        (1, max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    for timestep, token in enumerate(tokens):
        if token in input_features_dict:
            user_input_matrix[0, timestep, input_features_dict[token]] = 1.
    return user_input_matrix

# Method that will create a response using seq2seq model we built


def generate_response(user_input):
    input_matrix = string_to_matrix(user_input)
    chatbot_response = decode_response(input_matrix)
    # Remove <START> and <END> tokens from chatbot_response
    chatbot_response = chatbot_response.replace("<START>", '')
    chatbot_response = chatbot_response.replace("<END>", '')
    return chatbot_response


def make_exit(reply):
    for exit_command in exit_commands:
        if exit_command in reply:
            return True
    return False


def log(message):  # funcion de logging para heroku
    print(str(message))
    sys.stdout.flush()


if __name__ == '__main__':
    app.run(debug=True)
