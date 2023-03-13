import pickle

def save(G, path):

    # save graph object to file
    pickle.dump(G, open(path, 'wb'))
    print('Graph saved')

def read(path):
# load graph object from file
    return pickle.load(open(path, 'rb'))