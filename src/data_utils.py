from datetime import datetime
from torch_geometric.datasets import EllipticBitcoinDataset


start_time = None


def timer(start, model_name):
    global start_time
    if start:
        start_time = datetime.now()
    elif not start:
        if start_time is None:
            print("timer wasn't started")
        else:
            stop_time = datetime.now()
            length = stop_time - start_time
            print(f'The computing time for {model_name} is {length}')

def e_load():
    return EllipticBitcoinDataset(root='../data/Elliptic')

def e_drop_labels(data):
    e_labels = data.y.clone()  # Store labels for later evaluation
    data.y = None
    return data, e_labels
