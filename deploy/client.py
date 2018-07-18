# coding: utf-8

from urllib.request import urlopen
from urllib.parse import urlencode
import json


def similarity(question, relation):
    url = 'http://localhost:9000'
    params = {'question': question, 'relation': relation}
    params = bytes(json.dumps(params), encoding='utf-8')
    res = urlopen(url, params)
    if res.code == 200:
        post_data = eval(res.read().decode('utf-8').split('/')[-1])
        post_data['similarity'] = eval(post_data['similarity'])
        similarity = list()
        for s in post_data['similarity']['similarity']:
            similarity.append(s[0])

        return similarity


if __name__ == '__main__':
    question = '$ARG1 what does <e> people speak $ARG2'
    relation = ['location.country.languages_spoken', 'soccer.football_team_manager.team..soccer.football_team_management_tenure.to', 'law.invention.date_of_invention']
    simi = similarity(question, relation)
    print(simi)
