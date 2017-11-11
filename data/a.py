import pickle

f = open("validation.p", "rb")
e = pickle.load(f)
for l in e:
    print(l)
