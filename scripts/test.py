import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from omegaconf import OmegaConf

from dataloader import Dataloader
from model import EncoderRNN


conf = OmegaConf.load('./configs/train.yaml')

dataloader = Dataloader(conf.dataloader.path)


model = EncoderRNN(dataloader.n_letters, conf.model.hidden_layer_size, dataloader.n_categories)

model.load(conf.model.path)

# Just return an output given a line
def evaluate(line_tensor):
    hidden = model.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    return output

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(dataloader.lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, dataloader.all_categories[category_index]))
            predictions.append([value, dataloader.all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('sakura')





