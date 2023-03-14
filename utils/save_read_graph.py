import json
import pickle
from pathlib import Path

def save(G, args, path, network_name):
    # save graph object to file
    file_path = Path(path) / network_name
    network_path = file_path / 'network.gpickle'
    file_path.mkdir(parents=True, exist_ok=True)

    for k, v in args.items():
        arg_name = k + '.txt'
        arg_path = file_path / arg_name
        if type(v) == dict:
            v = {str(key): val for key, val in v.items() }
        with open(arg_path, 'w') as file:
            file.write(json.dumps(v)) # use `json.loads` to do the reverse

    pickle.dump(G, open(network_path, 'wb'))
    print('Graph saved')

def read(path):
    
    # load graph object from file
    file_path = Path(path)
    network_path = file_path / 'network.gpickle'
    x_path = file_path / 'station_size_coefficient.txt'
    y_path = file_path / 'demand_per_path_coefficient.txt'

    graph = pickle.load(open(network_path, 'rb'))
    x =  json.load(open(x_path, 'rb'))
    y = json.load(open(y_path, 'r'))
    return graph, x, y