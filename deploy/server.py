#!/usr/bin/env python3
"""
Very simple HTTP server in python for logging requests
Usage::
    ./server.py [<port>]
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import numpy as np
from keras.models import Model
import sys
import os
from io import BytesIO
import simplejson

sys.path.append('..')

from preprocess import process_one
from model_eval import model_construct

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_model(weight_path='../my_model_weights.h5'):
    model = model_construct()
    model.load_weights(weight_path)
    return model


def predict(model, question_feature, relation_feature, relation_all_feature):
    question_feature = np.array(question_feature)
    relation_feature = np.array(relation_feature)
    relation_all_feature = np.array(relation_all_feature)
    result = model.predict([question_feature, relation_feature, relation_all_feature], batch_size=1024)
    similarity = bytes(str(dict({'similarity': result.tolist()})), encoding='utf-8')
    return similarity


class S(BaseHTTPRequestHandler):

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = simplejson.loads(self.rfile.read(content_length)) # <--- Gets the data itself
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n", str(self.path), str(self.headers), post_data)

        self._set_response()
        question_feature = list()
        relation_feature = list()
        relation_all_feature = list()
        for i in range(len(post_data['relation'])):
            q_f, r_f, r_a_f = process_one(post_data['question'], post_data['relation'][i])
            question_feature.append(q_f)
            relation_feature.append(r_f)
            relation_all_feature.append(r_a_f)

        similarity = predict(model, question_feature, relation_feature, relation_all_feature)

        similarity = bytes(str(dict({'similarity': similarity})), encoding='utf-8')
        self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))
        self.wfile.write(similarity)


def run(server_class=HTTPServer, handler_class=S, port=9000):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')


if __name__ == '__main__':
    from sys import argv
    model = load_model()

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
