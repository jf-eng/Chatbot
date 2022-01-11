import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Get what is the available device for the dev env
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained responses so that we can access the responses
with open('/home/jfeng/projects/Python/personal/chatbot/src/train.json', 'r') as json_data:
    train = json.load(json_data)

# Loading up trained model
FILE = "data.pth" # file obtained after running train.py
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Instantiate model with same params and load the trained model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

if __name__ == "__main__":
    bot_name = "John"
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break
        
        # Can add other functionality here
        # elif sentence == "FAQ mode":
        #     pass
        # elif sentence == "Para mode":
        #     pass

        # Basic chatbot functionatility

        # Change the input sentence into acceptable format for model
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        # reformat X into acceptable input for the model
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        # Classify the input sentence into the tag
        output = model(X)
        _, predicted = torch.max(output, dim=1) # output from model, predicted is the index of the tag predicted

        tag = tags[predicted.item()] # index the predicted tag

        # Apply softmax now to get probability (ie confidence) of the output tag. In training, we used CrossEntropyLoss()
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()] # probability/confidence of the predicted tag
        if prob.item() > 0.75:
            for response in train['train']:
                if tag == response["tag"]:
                    # Take a random choice of response from train.json according to the predicted tag
                    print(f"{bot_name}: {random.choice(response['responses'])}")
        else:
            print(f"{bot_name}: I do not understand...")