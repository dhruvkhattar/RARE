from gensim.models import Doc2Vec, doc2vec
import pdb
from tqdm import tqdm
import json
import pickle
import os
from ast import literal_eval

def parse(path, lang):
    
    files = []
    for filename in os.listdir(path):
        files.append(filename)
    
    articles = {}

    for filename in tqdm(files):
        with open(path + '/' + filename) as fp:
            for line in fp:
                l = line.split('\t')
                if l[0] == 'an':
                    continue
                else:
                    art = json.loads(l[1])
                    try:
                        if art['lang'] == lang:
                            if not articles.has_key(l[0]):
                                try:
                                    cat = art['categories']
                                    for c in cat:
                                        if c not in articles:
                                            articles[c] = []
                                        articles[c].append(art['title'])
                                except:
                                    pass
                    except:
                        pass
    
    if not os.path.exists("../data/veooz/aug/" + lang):
        os.makedirs("../data/veooz/aug/" + lang)
    pickle.dump(articles, open('../data/veooz/aug/' + lang + '/articles_cat.pkl', 'w'))
    
if __name__ == '__main__':

    parse('../veooz_data/aug', 'ml')
