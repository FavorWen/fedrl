import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import pickle
from helper import LogSaver

logger = logging.getLogger('')
logger.setLevel(logging.INFO)

with open('train_data/seed3_archCNN_datasetCIFAR10_nums25_partiid.elog', 'rb') as f:
    ls = pickle.load(f)


print(ls.size)
print(ls.history_dim)

print(ls.history_pool)

for h in ls.history_pool:
    print(h)
