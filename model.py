from __future__ import division
import numpy as np
import random
import pdb
from keras.layers import Layer, Input, merge, Dense, LSTM, Bidirectional, GRU, SimpleRNN
from keras.layers.merge import concatenate, dot, multiply
from keras.models import Model
from attention import AttentionWithContext
from keras.callbacks import ModelCheckpoint
from create_inputs import creator
from tqdm import tqdm
import os
import pickle as pkl

class RARE:

    def __init__(self, rlg, nlayers, embedding_size_article, history):

        self.model = None
        self.rlg = rlg
        self.history = history
        self.layers = nlayers
        self.embedding_size_article = embedding_size_article


    def create_model(self):

        user_read = Input(shape=(self.history, self.embedding_size_article))
        user_case = Input(shape=(self.embedding_size_article, ))

        if self.rlg == 0:
            recurrent_layer = SimpleRNN(128, return_sequences=True)(user_read)
            recurrent_layer2 = SimpleRNN(128, return_sequences=False)(user_read)
        elif self.rlg == 1:
            recurrent_layer = LSTM(128, return_sequences=True)(user_read)
            recurrent_layer2 = LSTM(128, return_sequences=False)(user_read)
        else:
            recurrent_layer = GRU(128, return_sequences=True)(user_read)
            recurrent_layer2 = GRU(128, return_sequences=False)(user_read)
       
        attention_layer = AttentionWithContext()(recurrent_layer)
        concat_layer = concatenate([attention_layer, recurrent_layer2])

        if self.layers >= 1:
            left_layer = Dense(128, activation='relu')(concat_layer)
            right_layer = Dense(128, activation='relu')(user_case)
        if self.layers >= 2:
            left_layer = Dense(64, activation='relu')(left_layer)
            right_layer = Dense(64, activation='relu')(right_layer)
        if self.layers >= 3:
            left_layer = Dense(32, activation='relu')(left_layer)
            right_layer = Dense(32, activation='relu')(right_layer)

        elem_wise = multiply([left_layer, right_layer])

        output = Dense(1, activation='sigmoid')(elem_wise)


        self.model = Model(inputs=[user_read] + [user_case], outputs=output)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])


    def fit_model(self, inputs, output, pathname):
        if not os.path.exists("../weights/" + pathname):
            os.makedirs("../weights/" + pathname)
        filepath="../weights/"+pathname+"/weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(inputs, output, validation_split=0.2, epochs=50, callbacks=callbacks_list, verbose=1) 


    def get_model_summary(self):
        print self.model.summary()


def train(rlg, nlayers, rh, n_negs, pathname):
    c = creator(300, '../data/' + pathname + '/user_hist.pkl', '../data/' + pathname + '/user_key.pkl', '../data/' + pathname + '/item_key.pkl', '../data/' + pathname + '/art_embed.pkl', '../data/' + pathname + '/negs')
    user_read, pos_neg, truth, user_test, test, cold_user, cold_test = c.create_data_clef(10, 15, rh,  n_negs)
    model_test = RARE(rlg, nlayers, 300, rh)
    model_test.create_model()
    model_test.get_model_summary()
    if rlg == 0:
        pathname += '-rnn-'
    if rlg == 2:
        pathname += '-gru-'
    model_test.fit_model([user_read, pos_neg], truth, pathname + 'yo-' + str(rh) + '-' + str(nlayers) + '-' + str(n_negs))


def test(rlg, nlayers, rh, n_negs, pathname):
    c = creator(300, '../data/' + pathname + '/user_hist.pkl', '../data/' + pathname + '/user_key.pkl', '../data/' + pathname + '/item_key.pkl', '../data/' + pathname + '/art_embed.pkl', '../data/' + pathname + '/negs')
    user_read, pos_neg, truth, user_test, test, cold_user, cold_test = c.create_data_clef(10, 15, rh, n_negs)
    model_test = RARE(rlg, nlayers, 300, rh)
    model_test.create_model()
    model_test.get_model_summary()
    if rlg == 0:
        pathname += '-rnn-'
    if rlg == 2:
        pathname += '-gru-'
    pathname = pathname + '-' + str(rh) + '-' + str(nlayers) + '-' + str(n_negs)

    results = {}
    for filename in os.listdir('../weights/' + pathname):
        model_test.model.load_weights('../weights/' + pathname + '/' + filename)
        HR = []
        NDCG = []
        hr = [0]*10
        ndcg = [0]*10

        for user in tqdm(range(user_test.shape[0])):
            out = model_test.model.predict([np.array([user_test[user]]*100), np.array(test[user])])
            sorted_items = [i[0] for i in sorted(enumerate(out), key=lambda x:x[1])]
            sorted_items.reverse()
            for k in range(10):
                rec = sorted_items[:k+1]
                if 99 in rec:
                    hr[k] += 1
                for pos in range(k+1):
                    if rec[pos] == 99:
                        ndcg[k] += 1 / np.log2(1+pos+1)
        
        for k in range(10):
            print k, 'hr',  hr[k], 'ndcg', ndcg[k]
            HR.append(float(hr[k]) / float(user_test.shape[0]))
            NDCG.append(float(ndcg[k]) / float(user_test.shape[0]))
            print k, 'HR',  HR[k], 'NDCG', NDCG[k]
        results[filename] = [HR, NDCG]
    pkl.dump(results, open('../results/' + pathname, 'w'))


if __name__ == "__main__":
    #pathname = 'veooz/aug/id'
    pathname = 'clef'
    train(1, 2, 12, 2, pathname)
    #test(1, 2, 12, 2, pathname)
