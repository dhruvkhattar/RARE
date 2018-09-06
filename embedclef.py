from gensim.models import Doc2Vec, doc2vec
import pdb 
from tqdm import tqdm
import json
import pickle

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

    articles = json.load(open('../data/articles.json'))
    sents = []
    articleId = {}
    articleEmbedding = {}
    uid = 0 
    for article in tqdm(articles):
        articleId[article] = 'SENT_%s' % uid 
        sents.append(articles[article]['title'] + articles[article]['text'])
        uid += 1
    
    e = Embedding(sents)
    e.label()
    e.train()
    
    for k in tqdm(articleId):
        articleEmbedding[k] = e.model.docvecs[articleId[k]]

    fp = open('../data/article_embed.pkl', 'w')
    pickle.dump(articleEmbedding, fp)
