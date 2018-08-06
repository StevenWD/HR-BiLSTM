# HR-BiLSTM

Implementation for paper "Improved Neural Relation Detection for Knowledge Base Question Answering". The glove word embedding file can be download from [GloVe](https://nlp.stanford.edu/projects/glove/)



#### Train

Preprocess the training file and test file.

```python
python3 preprocess.py
```

Train model.

```python
python3 model.py
```

 

#### Test

Evaluate model with test file.

```python
python3 eval.py
```



#### Deploy

Packing models as a http service. Run commands in `deploy` folder.

```python
python3 server.py 9000
```

The function for calculate similarity is implement as `client.similarity`, use `question` and `relation` as input.



#### Config

Read and modify the config file `config.ini`
