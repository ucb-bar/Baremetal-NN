import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from omegaconf import OmegaConf

from dataloader import Dataloader
from model import EncoderRNN
from trainer import EncoderTrainer


conf = OmegaConf.load('./configs/eval.yaml')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader = Dataloader(conf.dataloader.path)
dataloader.load()

model = EncoderRNN(dataloader.n_letters, conf.model.hidden_layer_size, dataloader.n_categories)
model.load(conf.model.path)


trainer = EncoderTrainer(
    model=model, 
    criterion=None, 
    optimizer=None,
    logger=None,
    dataloader=dataloader)

@torch.no_grad()
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        print(dataloader.lineToTensor(input_line))
        output = trainer.evaluate(dataloader.lineToTensor(input_line))

        print("hidden shape", trainer.model.prev_hidden_state.shape)

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s id: %d' % (value, dataloader.all_categories[category_index], category_index))
            predictions.append([value, dataloader.all_categories[category_index]])

# predict("Dovesky")
# predict("Jackson")
predict("sakura")

