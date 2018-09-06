from gensim.models import Doc2Vec, doc2vec
import pdb
from tqdm import tqdm
import json
import pickle
import os
from ast import literal_eval

class Embedding():

    def __init__(self, sents):
        
        self.sents = sents
        self.labelledSents = []

    def label(self):
        for uid, line in enumerate(self.sents):
            self.labelledSents.append(doc2vec.LabeledSentence(words=line.split(), tags=['SENT_%s' % uid]))

    def train(self):
        self.model = Doc2Vec()
        self.model.build_vocab(self.labelledSents)
        for i in tqdm(range(10)):
            self.model.train(self.labelledSents)
        

if __name__ == '__main__':

    lang = 'te'
    articles = pickle.load(open('../data/veooz/aug/' + lang + '/articles.pkl'))

    sents = []
    articleId = {}
    articleEmbedding = {}
    uid = 0
    for article in tqdm(articles):
        articleId[article] = 'SENT_%s' % uid
        sents.append(articles[article])
        uid += 1
    
    e = Embedding(sents)
    e.label()
    e.train()
    e.model.save('../data/veooz/aug/' + lang + '/embed_model')
    
    for k in tqdm(articleId):
        articleEmbedding[k] = e.model.docvecs[articleId[k]]

    fp = open('../data/veooz/aug/' + lang + '/art_embed.pkl', 'w')
    pickle.dump(articleEmbedding, fp)
