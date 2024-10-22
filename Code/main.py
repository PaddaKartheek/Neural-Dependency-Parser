from model import ParserModel
from utils import Utils, Sentence_Parser
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import pickle

class MyDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]

def train(model, train_loader, num_epochs, dev_data, obj):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_input, batch_output in tqdm(train_loader, total = len(train_loader), desc="Training"):
            w_input, t_input, l_input = batch_input[0], batch_input[1], batch_input[2]
            optimizer.zero_grad()
            model_output = model(w_input, t_input, l_input)
            loss = criterion(model_output, batch_output)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_input[0].size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}")
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for sent in tqdm(dev_data, total=len(dev_data), desc = "Testing"):
                parser = Sentence_Parser(sent)
                parser.parse(obj, model)
                c, t =  parser.get_pred(obj)
                correct += c
                total += t
        print("UAS of Dev set: ", (correct / total) * 100.0)


def test(model, test_data, obj):
    model.eval()
    correct, total = 0, 0
    correct1, total1 = 0, 0
    with torch.no_grad():
        for sent in tqdm(test_data, total=len(test_data), desc = "Testing"):
            parser = Sentence_Parser(sent)
            parser.parse(obj, model)
            c, t = parser.get_pred(obj)
            c1, t1 = parser.get_pred(obj, UAS=False)
            correct += c
            total += t
            correct1 += c1
            total1 += t1
    print("UAS of Test set: ", (correct / total) * 100.0)
    print("LAS of Test set: ", (correct1 / total1) * 100.0)

if __name__ == "__main__":
    obj = Utils()
    #Preprocessing
    train_instances, test_data, dev_data, Ew = obj.preprocessing()

    Et = np.random.uniform(low=-0.01, high=0.01, size=(len(obj.tag2id), 50)).astype('float32')
    El = np.random.uniform(low=-0.01, high=0.01, size=(len(obj.rel2id), 50)).astype('float32')

    model = ParserModel(Ew, Et, El)
    train_inputs, train_outputs = train_instances[0], train_instances[1]
    num_epochs = 15
    learning_rate = 0.0005
    batch_size = 512
    reg_term = 1e-8

    model = ParserModel(Ew, Et, El)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    train_loader = DataLoader(dataset=MyDataset(train_inputs, train_outputs), batch_size=batch_size, shuffle=True)
    #Training
    train(model, train_loader, num_epochs, dev_data, obj)

    #Testing
    test(model, test_data, obj)
    torch.save(model.state_dict(), 'model.pt')
    with open('file.pkl', 'wb') as file:
        pickle.dump(obj, file)