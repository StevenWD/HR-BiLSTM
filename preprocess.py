# coding: utf-8

from configparser import ConfigParser
import numpy as np
import json
from tqdm import tqdm

config = ConfigParser()
config.read('./config.ini')

def readData(filepath):
    data = list()
    with open(filepath, 'r') as f:
        for line in f:
            one_data = line.strip('\n').split('\t')

            gold_relation = [int(num)-1 for num in one_data[0].split() if num.strip()]
            neg_relation = [int(num)-1 for num in one_data[1].split() if num.strip()]
            question = one_data[2].split()
            data.append([gold_relation, neg_relation, question])

    return data

def readRelation(filepath):
    relation = list()
    relation_all = list()
    with open(filepath, 'r') as f:
        for line in f:
            one_relation = line.strip('\n').split('.')
            one_relation_all = list()
            for r in one_relation:
                for w in r.split('_'):
                    one_relation_all.append(w)
            relation.append(one_relation)
            relation_all.append(one_relation_all)
    return relation, relation_all


def questionStat(data):
    word_dict = dict()
    word_dict['#UNK#'] = len(word_dict)

    for one_data in data:
        question = one_data[2]
        for word in question:
            if word_dict.get(word, -1) == -1:
                word_dict[word] = len(word_dict)

    return word_dict

def relationStat(relation):
    word_dict = dict()
    word_dict['#UNK#'] = len(word_dict)

    for one_relation in relation:
        for word in one_relation:
            if word_dict.get(word, -1) == -1:
                word_dict[word] = len(word_dict)

    return word_dict

def gloveEmbedding(embedding_filepath):
    all_word_embedding = dict()
    with open(embedding_filepath) as fin:
        for line in tqdm(fin):
            if line.strip():
                seg_res = line.split(" ")
                seg_res = [word.strip() for word in seg_res if word.strip()]

                key = seg_res[0]
                value = [float(word) for word in seg_res[1:]]
                all_word_embedding[key] = value
    return all_word_embedding


def questionEmbedding(question_words, all_word_embedding):

    reverse_question_words = dict()
    for key, value in question_words.items():
        reverse_question_words[str(value)] = key

    embedding_matrix = []
    for i in range(len(reverse_question_words)):
        i_str = str(i)
        key = reverse_question_words[i_str]
        value = all_word_embedding.get(key, -1)
        if value == -1:
            value = np.random.uniform(low=-0.5, high=0.5, size=(config.getint('pre', 'word_emd_length'),)).tolist()
        embedding_matrix.append(value)

    embedding_matrix = np.asarray(embedding_matrix)
    return embedding_matrix

def relationEmbedding(relation_words, all_word_embedding):
    reverse_relation_words = dict()
    for key, value in relation_words.items():
        reverse_relation_words[str(value)] = key

    embedding_matrix = []
    for i in range(len(reverse_relation_words)):
        i_str = str(i)
        key = reverse_relation_words[i_str]
        value = all_word_embedding.get(key, -1)
        if value == -1:
            value = np.random.uniform(low=-0.5, high=0.5, size=(config.getint('pre', 'relation_emd_length'), )).tolist()
        embedding_matrix.append(value)

    embedding_matrix = np.asarray(embedding_matrix)
    return embedding_matrix

def process(data, relation, relation_all, question_dict, relation_dict, relation_all_dict):
    question_feature = list()
    relation_feature = list()
    relation_feature_neg = list()
    relation_all_feature = list()
    relation_all_feature_neg = list()
    label = list()
    neg_number = list()
    
    for one_data in data:
        gold_relation = one_data[0]
        neg_relation = one_data[1]
        question = one_data[2]

        one_question_feature = np.zeros(config.getint('pre', 'question_maximum_length'))
        for index in range(min(config.getint('pre', 'question_maximum_length'), len(question))):
            word = question[index]
            if question_dict.get(word, -1) == -1:
                one_question_feature[index] = question_dict['#UNK#']
            else:
                one_question_feature[index] = question_dict[word]

        for one_relation in gold_relation:
            neg_number.append(len(neg_relation))
            one_relation_feature = np.zeros(config.getint('pre', 'relation_maximum_length'))
            one_relation_word = relation[one_relation]
            for index in range(min(config.getint('pre', 'relation_maximum_length'), len(one_relation_word))):
                word = one_relation_word[index]
                if relation_dict.get(word, -1) == -1:
                    one_question_feature[index] = relation_dict['#UNK#']
                else:
                    one_relation_feature[index] = relation_dict[word]

            one_relation_all_feature = np.zeros(config.getint('pre', 'relation_word_maximum_length'))
            one_relation_all_word = relation_all[one_relation]
            for index in range(min(config.getint('pre', 'relation_word_maximum_length'), len(one_relation_all_word))):
                word = one_relation_all_word[index]
                if relation_all_dict.get(word, -1) == -1:
                    one_relation_all_feature[index] = relation_all_dict['#UNK#']
                else:
                    one_relation_all_feature[index] = relation_all_dict[word]

            for _ in neg_relation:
                question_feature.append(one_question_feature)
                relation_feature.append(one_relation_feature)
                relation_all_feature.append(one_relation_all_feature)
                label.append([-1.0, 1.0])

        for _ in gold_relation:
            for one_relation in neg_relation:
                one_relation_feature = np.zeros(config.getint('pre', 'relation_maximum_length'))
                one_relation_word = relation[one_relation]
                for index in range(min(config.getint('pre', 'relation_maximum_length'), len(one_relation_word))):
                    word = one_relation_word[index]
                    if relation_dict.get(word, -1) == -1:
                        one_relation_feature[index] = relation_dict['#UNK#']
                    else:
                        one_relation_feature[index] = relation_dict[word]

                one_relation_all_feature = np.zeros(config.getint('pre', 'relation_word_maximum_length'))
                one_relation_all_word = relation_all[one_relation-1]
                for index in range(min(config.getint('pre', 'relation_word_maximum_length'), len(one_relation_all_word))):
                    word = one_relation_all_word[index]
                    if relation_all_dict.get(word, -1) == -1:
                        one_relation_all_feature[index] = relation_all_dict['#UNK#']
                    else:
                        one_relation_all_feature[index] = relation_all_dict[word]
                relation_feature_neg.append(one_relation_feature)
                relation_all_feature_neg.append(one_relation_all_feature)

    json.dump(neg_number, open('./neg_number.json', 'w'), indent=4)
    return question_feature, relation_feature, relation_all_feature, relation_feature_neg, relation_all_feature_neg, label

def process_one(question, relation, relation_all):
    question_dict = json.load(open('./question_dict.json', 'r'))
    relation_dict = json.load(open('./relation_dict.json', 'r'))
    relation_all_dict = json.load(open('./relation_all_dict.json', 'r'))

    question_feature = np.zeros(config.getint('pre', 'question_maximum_length'))
    for index in range(min(config.getint('pre', 'question_maximum_length'), len(question))):
        word = question[index]
        if question_dict.get(word, -1) == -1:
            question_feature[index] = question_dict['#UNK#']
        else:
            question_feature[index] = question_dict[word]

    relation_feature = np.zeros(config.getint('pre', 'relation_maximum_length'))
    relation_word = relation.split('.') 
    for index in range(min(config.getint('pre', 'relation_maximum_length'), len(relation_word))):
        word = relation_word[index]
        if relation_dict.get(word, -1) == -1:
            relation_feature[index] = relation_dict['#UNK#']
        else:
            relation_feature[index] = relation_dict[word]

    relation_all_feature = np.zeros(config.getint('pre', 'relation_word_maximum_length'))
    relation_all_word = []
    for r in relation_word:
        for rr in r.split('_'):
            relation_all_word.append(rr)
    for index in range(min(config.getint('pre', 'relation_word_maximum_length'), len(relation_all_word))):
        word = relation_all_word[index]
        if relation_all_dict.get(word, -1) == -1:
            relation_all_feature[index] = relation_all_dict['#UNK#']
        else:
            relation_all_feature[index] = relation_all_dict[word]

    return question_feature, relation_feature, relation_all_feature

def dump(prefix, question_feature, relation_feature, relation_all_feature, relation_feature_neg, relation_all_feature_neg, label):
    np.save(prefix+'question_feature.npy', question_feature)
    np.save(prefix+'relation_feature.npy', relation_feature)
    np.save(prefix+'relation_all_feature.npy', relation_all_feature)
    np.save(prefix+'relation_feature_neg.npy', relation_feature_neg)
    np.save(prefix+'relation_all_feature_neg.npy', relation_all_feature_neg)
    np.save(prefix+'label.npy', label)


if __name__ == '__main__':
    print('Embedding...')
    all_word_embedding = gloveEmbedding(config.get('pre', 'embedding_filepath'))
    print('Relations....')
    relation, relation_all = readRelation(config.get('pre', 'relation_filepath'))
    relation_dict = relationStat(relation)
    json.dump(relation_dict, open('relation_dict.json', 'w'))
    relation_all_dict = relationStat(relation_all)
    json.dump(relation_all_dict, open('relation_all_dict.json', 'w'))
    relation_emd_matrix = relationEmbedding(relation_dict, all_word_embedding)
    relation_all_emd_matrix = relationEmbedding(relation_all_dict, all_word_embedding)

    np.save('relation_emd_matrix.npy', relation_emd_matrix)
    np.save('relation_all_emd_matrix.npy', relation_all_emd_matrix)

    print('Data...')
    data = readData(config.get('pre', 'train_filepath'))
    question_dict = questionStat(data)
    json.dump(question_dict, open('question_dict.json', 'w'))
    question_emd_matrix = questionEmbedding(question_dict, all_word_embedding)
    np.save('question_emd_matrix.npy', question_emd_matrix)
    question_feature, relation_feature, relation_all_feature, relation_feature_neg, relation_all_feature_neg, label = process(data, relation, relation_all, question_dict, relation_dict, relation_all_dict)
    dump('train_', question_feature, relation_feature, relation_all_feature, relation_feature_neg, relation_all_feature_neg, label)

    data = readData(config.get('pre', 'test_filepath'))
    question_feature, relation_feature, relation_all_feature, relation_feature_neg, relation_all_feature_neg, label = process(data, relation, relation_all, question_dict, relation_dict, relation_all_dict)
    dump('test_', question_feature, relation_feature, relation_all_feature, relation_feature_neg, relation_all_feature_neg, label)

