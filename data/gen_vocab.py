import csv
import collections
data_path = "jd-comment.csv"
voca_path = "vocab.txt"

def dump_data(data):
    file = open(voca_path, "w")

    file.write('<unk> 0')
    file.write('\n')
    file.write('<pad> 0')
    file.write('\n')

    for l in data:
        file.write(l[0])
        file.write(' ')
        file.write(str(l[1]))
        file.write('\n')
    file.close()


def process_data(file):
    d = collections.defaultdict(int)
    for line in file:
        content = line["content"]
        for w in content:
            d[w] += 1
    print(d)
    return d

def read_data():
    with open(data_path) as f:
        f_csv = csv.DictReader(f)
        vocab = process_data(f_csv)
    data = sorted(vocab.items(), key=lambda d: d[1], reverse=True)
    dump_data(data)

read_data()
