import pickle
import torch
import numpy as np
from model import ParserModel
from utils import Sentence_Parser

with open('file.pkl', 'rb') as file:
    utils = pickle.load(file)



arg1 = np.random.rand(39549, 50).astype('float32')
arg2 = np.random.rand(47, 50).astype('float32')
arg3 = np.random.rand(41, 50).astype('float32')

model = ParserModel(arg1, arg2, arg3)
model.load_state_dict(torch.load('model.pt'))

dev_data = utils.get_data('../data/dev.conll')[4:10]
dev_data.pop(4)
v_dev_data = utils.vectorize(dev_data)
model.eval()
with torch.no_grad():
    for i, sent in enumerate(v_dev_data):
        parser = Sentence_Parser(sent)
        parser.parse(utils, model)
        print('='*20)
        print()
        print(dev_data[i]['words'])
        print()
        for h, d, t in parser.dependencies:
            print(dev_data[i]['words'][h-1], ' -> ', dev_data[i]['words'][d-1], ' : ', utils.id2tran[t][2:])
        print('='*20)





