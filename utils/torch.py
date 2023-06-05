import torch
import numpy as np

class EarlyStopping:
    """
    Early stopping to stop the training when
    the monitored metric does not improve after certain epochs.
    """
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0,
        verbose: bool = False,
        mode: str = 'max',
        baseline: float = None,
        start_from_epoch: int = 0,
        path: str = 'checkpoint.pt'
        ):
        """
        Initialize the early stopping object
        
        Parameters
        ----------
        patience : int, optional
            Number of epochs to wait before stopping the training.
            The default is 7.
        min_delta : float, optional
            Minimum change in the monitored quantity to qualify as
            an improvement.
            The default is 0.
        verbose : bool, optional
            If True, prints a message for each improvement.
            The default is False.
        mode : str, optional
            One of 'min' or 'max'. In 'min' mode, training will stop
            when the quantity monitored has stopped decreasing; in
            'max' mode it will stop when the quantity monitored has
            stopped increasing. The default is 'max'.
        baseline : float, optional
            Baseline value for the monitored quantity. Training will
            stop if the model doesn't show improvement over the baseline.
            The default is None.
        start_from_epoch : int, optional
            Epoch number from which to start counting the patience.
            The default is 0.
        path : str, optional
            Path to the checkpoint file.
            The default is 'checkpoint.pt'.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.start_from_epoch = start_from_epoch
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.metric_max = -np.Inf if mode == 'max' else np.Inf

    def check_improvement(self, metric):
        """
        Check if the monitored metric has improved
        and update the early stopping counter and checkpoint.
        
        Parameters
        ----------
        metric : float
            Monitored metric value to be compared.
        """
        if self.baseline is not None and self.mode == 'max':
            if metric >= self.baseline:
                if self.verbose:
                    print('Metric improved from baseline.')
                self.save_checkpoint(metric)
                self.counter = 0
                self.best_score = metric
            else:
                self.counter += 1
                if self.verbose:
                    print(f'Early stopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            if self.mode == 'max':
                score = metric
            else:
                score = -metric

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(metric)
            elif score < self.best_score + self.min_delta:
                if self.verbose:
                    print('Metric did not improve enough.')
                self.counter += 1
                if self.verbose:
                    print(f'Early stopping counter: {self.counter} out of {self.patience}')

                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                if self.verbose:
                    print('Metric improved.')
                self.best_score = score
                self.save_checkpoint(metric)
                self.counter = 0
                
    
    def save_checkpoint(self, metric):
        """
        Save the model checkpoint.
        
        Parameters
        ----------
        metric : float
            Monitored metric value to be saved.
        """
        if self.verbose:
            print(f'Metric increased ({self.metric_max:.6f} --> {metric:.6f}). Saving model...')
        torch.save(metric, self.path)
        self.metric_max = metric