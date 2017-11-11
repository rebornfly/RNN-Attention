import sys
import csv
sys.path.append("..")
import vocab as vc
import pickle
import collections

vocab_file = "vocab.txt"
train_file = "train.p"
valid_file = "validation.p"
score_file = "score.txt"

max_seq_length = 40
pad_id = 1

cab = vc.Vocab(vocab_file, verbose=False)
scores = {}
data = collections.defaultdict(list)

def load_scores():
    with open("final-score.csv") as f:
        f_csv = csv.DictReader(f)
        for i, row in enumerate(f_csv):
            scores[row['skuid']] = int(row['score']) - 5

def load_data():
    with open("jd-comment.csv", encoding = 'utf8') as f:
        f_csv = csv.DictReader(f)
        count = collections.defaultdict(int)
        for i, row in enumerate(f_csv):
            #统计评论长度
            l = len(row['content'])
            count[l] += 1
            #丢掉超过max length 的评论
            if len(row['content'] ) > max_seq_length:
                continue
            skuid = row['skuid']
            #丢掉没有标签的评论
            if skuid in scores.keys():
                continue

            data[skuid].append(row['content'])

        for d, k in data.items():
            print(d, len(k))

        print(sorted(count.items(), key=lambda d: d[1]))
        #for k, v in count.items():
        #    print(k, v)


def pre_process():
    train_data = {}
    valid_data = {}
    for k, v in data.items():
        lens = []
        sub_data= []
        for l in v:
            ids = []
            for w in l:
                id = cab.token_to_id(w)
                ids.append(id)
            ids += [pad_id]*(max_seq_length - len(ids))
            sub_data.append(ids)
            length = len(l)
            lens.append(length)
        tmp = [sub_data, lens]
        train_data[k] = tmp

    print("BEFORE SPLIT DATA ========>")
    for k, v in train_data.items():
        print("train data:", k, len(v[0]), len(v[1]))

    for k, v in train_data.items():
        size =  len(v[0])
        valid_size = int(size / 5)
        sub_valid  = v[0][0:valid_size]
        sub_length = v[1][0:valid_size]
        tmp = [sub_valid, sub_length]
        valid_data[k] = tmp

        v[0] = v[0][valid_size:]
        v[1] = v[1][valid_size:]

    print("AFTER SPLIT DATA ========>")
    for k, v in train_data.items():
        print("train data:", k, len(v[0]), len(v[1]))
    for k, v in valid_data.items():
        print("valid data:", k, len(v[0]), len(v[1]))
    dump_data(valid_data, valid_file)
    dump_data(train_data, train_file)

def dump_data(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

load_scores()
load_data()
pre_process()
