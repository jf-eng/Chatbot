import json
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import nltk
from nltk_utils import tokenize, stem, bag_of_words
# nltk.download('stopwords') only need to run once
from nltk.corpus import stopwords 

from model import NeuralNet

# Extract training data from train.json
with open("/home/jfeng/projects/Python/personal/chatbot/src/train.json", 'r') as f:
    train_data = json.load(f)

all_words, tags, xy = [], [], []

# Extract data from json and keep track of the words used in input sentences (represented by all_words)
for train in train_data['train']:
    tag = train['tag']
    tags.append(tag)
    for train_sentence in train['input']:
        token_sentence = tokenize(train_sentence)
        all_words += [stem(w) for w in token_sentence if w not in (stopwords.words('english') + ['!', '.', '?'])]
        xy.append((token_sentence, tag))
        
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Generating X_train, y_train
X_train, y_train = [], []
for input_sentence, tag in xy:
    bag = bag_of_words(input_sentence, all_words)
    X_train.append(bag)

    tag_idx = tags.index(tag)
    y_train.append(tag_idx)
    
X_train = np.array(X_train)
y_train = np.array(y_train)

# Inherit class Dataset from torch
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # allows for len() to see length of dataset
    def __len__(self):
        return self.n_samples

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

# Save model as data.pth
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')