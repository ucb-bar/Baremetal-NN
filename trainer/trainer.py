import io
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dataloader import Dataloader
from model import RNN
# from logger import Logger

class EncoderTrainer:
    def __init__(self, model: RNN, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, logger, dataloader: Dataloader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.dataloader = dataloader

    def step(self, category_tensor, line_tensor):
        self.model.initializeHiddenState()

        for i in range(line_tensor.size()[0]):
            output = self.model(line_tensor[i])

        loss: torch.Tensor = self.criterion.forward(output, category_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return output, loss.item()

    def evaluate(self, line_tensor: torch.Tensor):
        self.model.initializeHiddenState()

        for i in range(line_tensor.size()[0]):
            output = self.model(line_tensor[i])

        return output

    def train(self, n_iters):
        for step in range(n_iters+1):
            category, line, category_tensor, line_tensor = self.dataloader.getRandom()
            output, loss = self.step(category_tensor, line_tensor)
    
            if step % self.logger.log_every == 0:
                self.logger.log(step, loss)
                self.generateLoggerData(step)
                self.model.save(os.path.join(self.logger.logdir, "model_{steps}.pt".format(steps=step)))

    def generateLoggerData(self, step):
        # Keep track of correct guesses in a confusion matrix
        confusion = torch.zeros(self.dataloader.n_categories, self.dataloader.n_categories)
        n_confusion = 10000

        n_correct = 0
        log_text = ""

        # Go through a bunch of examples and record which are correctly guessed
        for i in range(n_confusion):
            category, line, category_tensor, line_tensor = self.dataloader.getRandom()
            
            output = self.evaluate(line_tensor)
            
            category_i = self.dataloader.all_categories.index(category)
            guess, guess_i = self.dataloader.categoryFromOutput(output)
            confusion[category_i][guess_i] += 1

            if category == guess:
                n_correct += 1
            
            if i < 5:
                log_text += "guess {line} - {guess}: {is_correct}\r\n\r\n".format(
                    line=line, 
                    guess=guess, 
                    is_correct="✓" if guess == category else "✗ (category)".format(category)
                )
        
        self.logger.logPredicted(step, log_text)
        self.logger.logAccuracy(step, n_correct / n_confusion)

        # Normalize by dividing every row by its sum
        for i in range(self.dataloader.n_categories):
            confusion[i] = confusion[i] / confusion[i].sum()

        # Set up plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusion.numpy())
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([""] + self.dataloader.all_categories, rotation=90)
        ax.set_yticklabels([""] + self.dataloader.all_categories)

        # Force label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        
        # Convert PNG buffer to TF image
        self.logger.logConfusionMatrix(step, fig)
        plt.close(fig)








class DecoderTrainer:
    def __init__(self, model, criterion, optimizer, logger, dataloader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.dataloader = dataloader


    def step(self, category_tensor, input_line_tensor, target_line_tensor):
        target_line_tensor.unsqueeze_(-1)
        hidden = self.model.initHidden()

        self.optimizer.zero_grad()
        loss = torch.Tensor([0])

        for i in range(input_line_tensor.size()[0]):
            output, hidden = self.model.forward(category_tensor, input_line_tensor[i], hidden)
            l = self.criterion(output, target_line_tensor[i])
            loss += l

        loss.backward()
        self.optimizer.step()

        return output, loss.item() / input_line_tensor.size(0)

    def train(self, n_iters):
        for step in range(1, n_iters + 1):
            print(step)
            category_tensor, input_line_tensor, target_line_tensor = self.dataloader.randomTrainingExample2()
            output, loss = self.step(category_tensor, input_line_tensor, target_line_tensor)
            
            
            self.logger.add_scalar("loss", loss, step)

