# coding: utf-8

from urllib.request import urlopen
from urllib.parse import urlencode
import json

def similarity(question, relation):
    url = 'http://localhost:9000'
    params = {'question': question,
            'relation': relation}
    params = bytes(json.dumps(params), encoding='utf-8')
    res = urlopen(url, params)
    if res.code == 200:
        simi = eval(res.read().decode('utf-8').split('/')[1])
        return simi['similarity']

if __name__ == '__main__':
    question = '$ARG1 what does <e> people speak $ARG2'
    relation = 'location.country.languages_spoken'
    simi = similarity(question, relation)
    print(simi)
