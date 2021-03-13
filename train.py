import json
from  nltk_utils import tokenise,stem,bag_of_words
import numpy as np
import torch
import torch.nn as nn
from model import NeuralNet
from torch.utils.data import Dataset,DataLoader

with open('intents.json','r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag  = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenise(pattern)
        all_words.extend(w)
        xy.append((w,tag))

ignore_words = ['?',',','!',';','.']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []

for (pattern,tag) in xy:
    bag = bag_of_words(pattern,all_words)
    X_train.append(bag)

    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

class ChatDataSet(Dataset):   
    def __init__(self):
        self.nb_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train
    
    def __getitem__(self,index):
        return (self.x_data[index],self.y_data[index])
    
    def __len__(self):
        return self.nb_samples

#HyperParameters
batch_size = 8
hidden_size = 8
input_size = len(all_words)
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataSet()
train_loader = DataLoader(dataset = dataset,batch_size=batch_size,shuffle=True,num_workers=2)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size,output_size)

#Loss and Optimiser
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters , lr = learning_rate)


for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        #words = words.to(device)
        #labels = labels.to(device)

        #forward
        outputs = model(words)
        loss = criterion(outputs,labels)

        #backward and optimiser step
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
    if (epoch + 1 )%100 == 0 :
        print(f'epoch {epoch+1} / {num_epochs}, loss = {loss.item():.4f} ')
print(f'Final Loss  = {loss.item():.4f} ')