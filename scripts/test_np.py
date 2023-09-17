import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import torch

from dataloader import Dataloader
from model import EncoderRNN_NP


dataloader = Dataloader()

n_hidden = 32
model = EncoderRNN_NP(dataloader.n_letters, n_hidden, dataloader.n_categories)

model.load("./logs/model.json")

# Just return an output given a line
def evaluate(line_tensor):
    hidden = np.zeros((1, n_hidden))
    
    for i in range(line_tensor.shape[0]):
        output, hidden = model.forward(line_tensor[i], hidden)
    return output

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(dataloader.lineToNumpy(input_line))
        output = output[0]

        # Get top N categories
        topi = np.argmax(output)
        topv = output[topi]

        value = topv
        category_index = topi
        print('(%.2f) %s' % (value, dataloader.all_categories[category_index]))



print(dataloader.all_letters)
quit()

predict('sakura')
predict('Jackson')

predict('Dovesky')




