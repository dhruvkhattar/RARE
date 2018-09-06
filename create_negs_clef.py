import pickle
import json
import random

def parser(path, x1, x2):
    data = pickle.load(open(path + '/user_hist.pkl'))

    lister = []
    idx = {}
    count = 0
    for each in data.keys():
        if len(data[each]) >= x1 and len(data[each]) <= x2:
            lister.append(each)
            idx[each] = count
            count += 1

    article_map = {}
    count = 0
    article_list = []

    for each in lister:
        for articles in data[each]:
            if not article_map.has_key(articles):
                article_map[articles] = count
                count += 1

    fp = open(path + '/negs', 'w')
    for each in idx.keys():
        neg_list = random.sample(article_map.keys(), 99)
        line = "(" + str(each) + ")" + "\t"
        for every in neg_list[:-1]:
            line += str(article_map[every]) + "\t"
        line += str(article_map[neg_list[-1]]) + "\n"
        fp.write(line)
    fp.close()

    pickle.dump(idx, open(path + '/user_key.pkl', 'w'))
    pickle.dump(article_map, open(path + '/item_key.pkl', 'w'))



def main(filename, x1, x2):
    parser(filename, x1, x2)

if __name__ == "__main__":
    main("../../radssm/data/clef", 10, 15)
