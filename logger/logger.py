import datetime
import time
import math
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir=None, print_every=5000, plot_every=1000):
        # Keep track of losses for plotting
        self.current_loss = 0
        self.all_losses = []

        if log_dir is None:
            self.log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.total_iter = 10000
        self.print_every = print_every
        self.plot_every = plot_every
        self.start = time.time()
        self.writer = SummaryWriter(self.log_dir)

    def timeSince(self, since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    
    def log(self, itr, loss, guess, guess_i, category, line):
        self.current_loss += loss
        
        self.writer.add_scalar("Loss/train", loss, itr)

        # Print ``iter`` number, loss, name and guess
        if itr % self.print_every == 0:
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (itr, itr / self.total_iter * 100, self.timeSince(self.start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if itr % self.plot_every == 0:
            self.all_losses.append(self.current_loss / self.plot_every)
            self.current_loss = 0

    def add_scalar(self, tag, scalar_value, step):
        if step % self.plot_every == 0:
            self.writer.add_scalar(tag, scalar_value, step)