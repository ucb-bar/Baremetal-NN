import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from dataloader import Dataloader
from model import DecoderRNN
from logger import Logger
from trainer import DecoderTrainer


conf = OmegaConf.load('./configs/train.yaml')

dataloader = Dataloader(conf.dataloader.path)
model = DecoderRNN(dataloader.n_categories, dataloader.n_letters, conf.model.hidden_layer_size, dataloader.n_letters)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=conf.trainer.learning_rate)
logger = Logger()

trainer = DecoderTrainer(
    model=model, 
    criterion=criterion, 
    optimizer=optimizer,
    logger=logger,
    dataloader=dataloader)


trainer.train(conf.trainer.n_iters)

model.save(conf.model.path)

max_length = 20

# Sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = dataloader.categoryTensor(category)
        input = dataloader.inputTensor(start_letter)
        hidden = model.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = model.forward(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == dataloader.n_letters - 1:
                break
            else:
                letter = dataloader.all_letters[topi]
                output_name += letter
            input = dataloader.inputTensor(letter)

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

# samples('Russian', 'RUS')

# samples('German', 'GER')

# samples('Spanish', 'SPA')


for i in range(10):
    samples('Chinese', 'CHI')


