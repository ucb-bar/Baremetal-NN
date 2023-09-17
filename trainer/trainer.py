import io

import numpy as np
import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import PIL.Image

class EncoderTrainer:
    def __init__(self, model, criterion, optimizer, logger, dataloader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.dataloader = dataloader

    def step(self, category_tensor, line_tensor):
        hidden = self.model.initHidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = self.model(line_tensor[i], hidden)

        loss = self.criterion(output, category_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return output, loss.item()

    def train(self, n_iters):
        for step in range(1, n_iters + 1):
            category, line, category_tensor, line_tensor = self.dataloader.randomTrainingExample()
            output, loss = self.step(category_tensor, line_tensor)
            
            guess, guess_i = self.dataloader.categoryFromOutput(output)
            
            self.logger.log(step, loss, guess, guess_i, category, line)

      
      
            if step % self.logger.plot_every == 0:

                # Keep track of correct guesses in a confusion matrix
                confusion = torch.zeros(self.dataloader.n_categories, self.dataloader.n_categories)
                n_confusion = 10000

                # Just return an output given a line
                def evaluate(line_tensor):
                    hidden = self.model.initHidden()

                    for i in range(line_tensor.size()[0]):
                        output, hidden = self.model.forward(line_tensor[i], hidden)

                    return output

                # Go through a bunch of examples and record which are correctly guessed
                for i in range(n_confusion):
                    category, line, category_tensor, line_tensor = self.dataloader.randomTrainingExample()
                    output = evaluate(line_tensor)
                    guess, guess_i = self.dataloader.categoryFromOutput(output)
                    category_i = self.dataloader.all_categories.index(category)
                    confusion[category_i][guess_i] += 1

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
                self.logger.writer.add_figure('confusion_matrix', fig, step)
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

