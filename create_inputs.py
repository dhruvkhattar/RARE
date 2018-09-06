import random
import numpy as np
import pdb
import json
import pickle as pkl

class creator:

    def __init__(self, embed_size, user_history, user_key, item_key, content_embed, neg_file):

        self.embed_size = embed_size
        self.user_read = pkl.load(open(user_history))
        self.user_map = pkl.load(open(user_key))
        self.article_map_rev = pkl.load(open(item_key))
        self.article_map = {v: k for k, v in self.article_map_rev.iteritems()}
        self.embed = pkl.load(open(content_embed))
        self.neg_file = open(neg_file)
        self.negs = self.neg_file.readlines()

    def create_data_clef(self, low, high, inp_size, n_negs):

        self.read_list = []
        self.pos_neg = []
        self.truth = []
        self.test = []
        self.user_test = []
        articles = []
       
        test = []
        for each in self.user_read.keys():
            if len(self.user_read[each]) < low or len(self.user_read[each]) > high:
                continue
            articles += self.user_read[each][:-1]
            test.append(self.user_read[each][-1])
            total_hist = map(lambda x: self.embed[str(x)], self.user_read[each])
            
            size = len(total_hist)
            read_hist = []
            
            article_array = []
            negs = self.negs[self.user_map[each]].split('\t')[1:]
            negatives = []
            for article in negs:
                negatives.append(self.article_map[int(article)])
                article_array.append(np.asarray(self.embed[str(negatives[-1])]))
            article_array.append(self.embed[str(self.user_read[each][-1])])
            
            self.test.append(article_array)

            if size > inp_size - 1:
                n_pos = size - inp_size - 1 
                read_hist = total_hist[:inp_size]
            else:
                n_pos = 1 
                read_hist = []
                padding = np.zeros(self.embed_size)
                for j in range(inp_size - size + 2): 
                    read_hist.append(padding)
                for j in range(size - 2): 
                    read_hist.append(total_hist[j])

            read_hist = np.asarray(read_hist)
            self.user_test.append(read_hist)
    
            for j in range(n_pos):
                self.read_list.append(read_hist)
                self.pos_neg.append(total_hist[size-2-j])
                self.truth.append(1)

            for j in range(n_pos*n_negs):
                selection = random.choice(negatives)
                self.read_list.append(read_hist)
                self.pos_neg.append(np.asarray(self.embed[str(selection)]))
                self.truth.append(0)

        self.cold_user = []
        self.cold_test = []

        ct = 0
        for i in range(len(self.test)):
            if test[i] not in articles:
                ct += 1
                self.cold_user.append(self.user_test[i])
                self.cold_test.append(self.test[i])
        
        print 'Cold Users:', ct

        self.read_list = np.asarray(self.read_list)
        self.pos_neg = np.asarray(self.pos_neg)
        self.truth = np.asarray(self.truth)
        self.test = np.asarray(self.test)
        self.user_test = np.asarray(self.user_test)
        self.cold_test = np.asarray(self.cold_test)
        self.cold_user = np.asarray(self.cold_user)

        return self.read_list, self.pos_neg, self.truth, self.user_test, self.test, self.cold_user, self.cold_test

    
    def create_data(self, inp_size, shift, n_negs):
        
        self.read_list = []
        self.pos_neg = []
        self.truth = []
        self.test = []
        self.user_test = []
    
        for each in self.user_read.keys():

            if len(self.user_read[each]) <= 1:
                continue

            article_array = []
            negs = self.negs[self.user_map[each]].split('\t')[1:]
            negatives = []
            for article in negs:
                negatives.append(self.article_map[int(article)])
                article_array.append(np.asarray(self.embed[negatives[-1]]))
            article_array.append(self.embed[self.user_read[each][-1]])
            
            self.test.append(article_array)
            self.user_read[each] = self.user_read[each][:-1]
            tsize = len(self.user_read[each])
            
            article_array = []
            if inp_size > tsize:
                for i in range(inp_size-tsize):
                    article_array.append(np.zeros_like(self.embed[str(self.user_read[each][0])]))
                for i in range(tsize):
                    article_array.append(self.embed[str(self.user_read[each][i])])
            else:
                for i in range(inp_size):
                    article_array.append(self.embed[str(self.user_read[each][-inp_size+i])])

            self.user_test.append(article_array)

            start = 0
            while tsize > 1:
                user_read = self.user_read[each][start:start+inp_size+1]
                if tsize > inp_size+1: 
                    size = inp_size+1
                else:
                    size = tsize
                tsize -= shift
                start += shift
                pad = inp_size + 1 - size
                pos_examples = 1 
    
                article_array = []
                for i in range(pad):
                    article_array.append(np.zeros_like(self.embed[str(user_read[0])]))

                for i in range(size - 1): 
                    article_id = str(user_read[i])
                    article_embed = np.asarray(self.embed[str(article_id)])
                    article_array.append(article_embed)
    
                self.read_list.append(article_array)
                self.pos_neg.append(np.asarray(self.embed[str(user_read[size-1])]))
                self.truth.append(1)

                for j in range(n_negs):
                    selection = random.choice(negatives)
                    self.read_list.append(article_array)
                    self.pos_neg.append(np.asarray(self.embed[str(selection)]))
                    self.truth.append(0)

        self.read_list = np.asarray(self.read_list)
        self.pos_neg = np.asarray(self.pos_neg)
        self.truth = np.asarray(self.truth)
        self.test = np.asarray(self.test)
        self.user_test = np.asarray(self.user_test)

        return self.read_list, self.pos_neg, self.truth, self.user_test, self.test
