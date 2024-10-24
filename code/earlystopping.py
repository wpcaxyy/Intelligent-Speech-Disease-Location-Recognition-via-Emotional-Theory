import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, acc,uacc, model,path):

        score = -acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(acc,uacc, model,path)
            return True
        elif score > self.best_score - self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(acc,uacc, model,path)
            self.counter = 0
            return True

    def save_checkpoint(self, acc,uacc, model,path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Test Acc Increased ({self.val_loss_min:.6f} --> {acc:.6f}).  Saving model ...')
            print(f'Test Acc : {acc}')
            print(f'Test U-Acc : {uacc}')
        torch.save(model.state_dict(), path)
        self.val_loss_min = acc