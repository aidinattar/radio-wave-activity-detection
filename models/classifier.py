"""
classifier.py

This file contains the model class, which is used
to create the model and train it.

TODO:
    - Add missing parts
    - Check if the code is correct
    - Add possible plots and results
"""
import os, sys
import torch
import numpy             as np
import pandas            as pd
import seaborn           as sns
import matplotlib.pyplot as plt
#import torchvision.utils as vutils

from torch.utils.data          import DataLoader, random_split
from torch.optim               import SGD, Adam, NAdam
from torch.nn                  import CrossEntropyLoss
from preprocessing.dataset     import Dataset
from exceptions                import OptionIsFalseError, WorkToDoError
from models.cnn_rd             import cnn_rd
from models.cnn_md             import cnn_md
from tqdm                      import tqdm
from sklearn.metrics           import confusion_matrix, accuracy_score,\
                                      precision_recall_fscore_support,\
                                      roc_curve, roc_auc_score,\
                                      precision_recall_curve, average_precision_score,\
                                      classification_report, auc
from utils                     import plotting, augmentation
from torchsummary              import summary
from torch.utils.data          import Subset
from memory_profiler           import profile
from sklearn.preprocessing     import label_binarize
#from torch.utils.tensorboard import SummaryWriter

fig_dir = 'figures'

class model(object):
    """
    Class to create the model
    """
    train_test_split_done = False
    model_created = False
    optimizer_created = False
    loss_created = False
    model_trained = False

    def __init__(
        self, 
        data: Dataset,
        case: int=0,
        model_type: str='CNN-MD',
        channels: int=1,
        device= torch.device('cuda' if torch.cuda.is_available() else 'cpu'),    
    ):
        """
        Constructor

        Parameters
        ----------
        data : Dataset
            Dataset object.
        case : int, optional
            Case to use. The default is 0.
            Possible values are:
                0: train and test using only one radar and one channel
                1: train on one radar and test on the other
                2: train and test on all the data
                3: train and test one radar per channel
        model_type : str, optional
            Type of model to create. Possible values are:
            'CNN-MD', 'CNN-RD'. The default is 'CNN-MD'.
        device : str, optional
            Device to use. The default is 'cuda'.
            Possible values are:
                'cpu': CPU
                'cuda': GPU
        """
        # Get the data
        self.data = data

        # Get the case
        self.case = case

        # Get the model type
        self.model_type = model_type
        
        # Get the device
        self.device = device
        
        # Get the number of channels
        assert channels in [1, 2], "The number of channels must be 1 or 2"
        self.channels = channels
        
        # Add tensorboard writer
        #self.writer = SummaryWriter()


    # TODO: add also the option to use the validation set
    def train_test_split(self,
                         test_size: float=0.2,
                         random_state: int=42):
        '''
        Split the data into training and testing data

        Parameters
        ----------
        test_size : float, optional
            Size of the test data. The default is 0.2.
        random_state : int, optional
            Random state. The default is 42.
        '''
        # define the random generator
        generator = torch.Generator().manual_seed(random_state)
        # Split the data
        if self.case == 0:            
            test_dim = int(test_size*len(self.data))
            train_dim = len(self.data) - test_dim
            self.train_data, self.test_data = random_split(self.data, [train_dim, test_dim], generator=generator)                 
        elif self.case == 1:    #this in standby
            self.train_data = self.data[0]
            self.test_data = self.data[1]
        elif self.case == 2:    #this in standby
            test_dim = int(test_size*len(self.data))
            train_dim = len(self.data) - test_dim
            self.train_data, self.test_data = random_split(self.data, [train_dim, test_dim], generator=generator)
        elif self.case == 3:
            test_dim = int(test_size*len(self.data))
            train_dim = len(self.data) - test_dim
            self.train_data, self.test_data = random_split(self.data, [train_dim, test_dim], generator=generator)
        else:
            raise ValueError('Invalid case')
        
        self.input_size = self.train_data[0][0].shape

        # Set the flag
        self.train_test_split_done = True


    def create_DataLoaders(self, batch_size: int=32, shuffle: bool=True, num_workers: int=0):
        """
        Create the data loaders
        
        Parameters
        ----------
        batch_size : int, optional
            Batch size. The default is 32.
        shuffle : bool, optional
            Shuffle the data. The default is True.
        num_workers : int, optional
            Number of workers. The default is 0.
        """
                  
        # Create the data loaders
        self.train_loader = DataLoader(self.train_data,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=num_workers)
        self.test_loader = DataLoader(self.test_data,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=num_workers)


    def augmentation(self, method=['time-mask'], **kwargs):
        """
        Augment the data.

        Parameters
        ----------
        method : list, optional
            List of methods to use.
            Possible values are:
                'resample', 'time-mask', 'doppler-mask', 'time-doppler-mask'
            The default is ['time-mask'].
        **kwargs : TYPE
            Keyword arguments to pass to the augmentation function.
            
        Raises
        ------
        ValueError
            Invalid data type.
        """
        if self.data.type=='rdn':
            self.augmentation_rdn(method=method, **kwargs)
        elif self.data.type=='mDoppler':
            self.augmentation_mDoppler(method=method, **kwargs)
        else:
            raise ValueError('Invalid data type')
        
        # Remove duplicates
        self.train_data.dataset.drop_duplicates()
        self.train_data.indices = np.arange(len(self.train_data.dataset))


    def augmentation_rdn(self, method=['time-mask'], **kwargs):
        """
        Augment the rdn data

        Parameters
        ----------
        method : list, optional
            List of methods to use.
        **kwargs : TYPE
            Keyword arguments to pass to the augmentation function.
        """

        if 'resample' in method:
            try:
                n_samples = kwargs['n_samples']
            except KeyError:
                n_samples = 5
            self.train_data = augmentation.resample(self.train_data,
                                                    data_dir='DATA_preprocessed',
                                                    data_file='data_cutted.npz',
                                                    data_type='rdn',
                                                    len_default=40,
                                                    n_samples=n_samples)
            
        if 'time-mask' in method:
            self.train_data = augmentation.time_mask(self.train_data)

        if 'doppler-mask' in method:
            self.train_data = augmentation.doppler_mask(self.train_data)
        
        if 'time-doppler-mask' in method:
            self.train_data = augmentation.time_doppler_mask(self.train_data)


    def augmentation_mDoppler(self, method=['time-mask'], **kwargs):
        """
        Augment the mDoppler data
        
        Parameters
        ----------
        method : list, optional
            List of methods to use.
        **kwargs : TYPE
            Keyword arguments to pass to the augmentation function.
        """
        if 'resample' in method:
            try:
                n_samples = kwargs['n_samples']
            except KeyError:
                n_samples = 5
            self.train_data = augmentation.resample(self.train_data,
                                                    data_dir='DATA_preprocessed',
                                                    data_file='data_cutted.npz',
                                                    data_type='mDoppler',
                                                    len_default=40,
                                                    n_samples=n_samples)
            
        if 'time-mask' in method:
            self.train_data = augmentation.time_mask(self.train_data)
            
        if 'doppler-mask' in method:
            self.train_data = augmentation.doppler_mask(self.train_data)
            
        if 'time-doppler-mask' in method:
            self.train_data = augmentation.time_doppler_mask(self.train_data)
            
        if 'time-doppler-mask' in method:
            self.train_data = augmentation.time_doppler_mask(self.train_data)
        

    def create_model(self, **kwargs):
        """
        Create the model
        
        Parameters
        ----------
        **kwargs : TYPE
            Keyword arguments to pass to the model class.
        """
        # call cnn_rd or cnn_md class
        if self.model_type == 'CNN-MD':
            if self.data.TYPE != 'mDoppler':
                OptionIsFalseError('do_mDoppler')
            self.model = cnn_md(
                **kwargs
            )
        elif self.model_type == 'CNN-RD':
            if self.data.type != 'rdn':
                OptionIsFalseError('do_rdn')
            self.model = cnn_rd(
                **kwargs
            )
        else:
            raise ValueError('Invalid model type')

        # Set the flag
        self.model_created = True


    def create_optimizer(self,
                         optimizer: str='SGD',
                         lr: float=0.01,
                         momentum: float=0.9,
                         weight_decay: float=0.0005,
                         nesterov: bool=True):
        """
        Create the optimizer

        Parameters
        ----------
        optimizer : str, optional
            Optimizer to use. The default is 'SGD'.
            Possible values are:
                'SGD': Stochastic Gradient Descent
                'Adam': Adam optimizer
        lr : float, optional
            Learning rate. The default is 0.01.
        momentum : float, optional
            Momentum. The default is 0.9.
        weight_decay : float, optional
            Weight decay. The default is 0.0005.
        nesterov : bool, optional
            Nesterov. The default is True.
        """
        if optimizer == 'SGD':
            self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=momentum)
        elif optimizer == 'Adam':
            if not nesterov:
                self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                self.optimizer = NAdam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Invalid optimizer')

        # Set the flag
        self.optimizer_created = True


    def create_loss(self, loss: str='CrossEntropyLoss'):
        """
        Create the loss function

        Parameters
        ----------
        loss : str, optional
            Loss function to use. The default is 'CrossEntropyLoss'.
            Possible values are:
                'CrossEntropyLoss': Cross Entropy Loss
                
        Raises
        ------
        ValueError
            Invalid loss function.
        """
        if loss == 'CrossEntropyLoss':
            self.loss = CrossEntropyLoss()
        #elif loss == :
            #self.loss = BCEWithLogitsLoss()
        else:
            raise ValueError('Invalid loss function')

        # Set the flag
        self.loss_created = True


    ######################################
    ######           CHECK         #######
    ######################################
    @profile
    def train_model(self,
                    epochs: int=10,
                    checkpoint: bool=False,
                    checkpoint_path: str='checkpoint.pt',
                    checkpoint_dir: str='checkpoints'
                    ):
        """
        Train the model

        Parameters
        ----------
        epochs : int, optional
            Number of epochs. The default is 10.
        checkpoint : bool, optional
            Save the model after every epoch. The default is False.
        checkpoint_path : str, optional
            Path to save the model. The default is 'checkpoint.pt'.

        Raises
        ------
        WorkToDoError
            If the train_test_split, create_model, create_optimizer,
            create_loss methods have not been called
        """
        if not self.train_test_split_done:
            raise WorkToDoError('train_test_split_done')
        if not self.model_created:
            raise WorkToDoError('model_created')
        if not self.optimizer_created:
            raise WorkToDoError('optimizer_created')
        if not self.loss_created:
            raise WorkToDoError('loss_created')

        # Use GPU
        self.model = self.model.to(self.device)

        # Define first best loss
        best_loss = np.inf

        # Lists to store the losses and accuracies
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        # Train the model
        for epoch in range(epochs):
            self.model.train()
            print(f'Epoch {epoch+1}/{epochs}')
            iterator = tqdm(self.train_loader)
            preds, targets = [], []
            for batch in iterator:
                # Get the data
                if self.channels==1:
                    data = batch[0].unsqueeze(1).to(self.device) #### batch['data'].to(device)
                else:
                    data = batch[0].to(self.device)
                target = batch[1].to(self.device) #### batch['target'].to(device)

                # Forward pass
                output = self.model(data)
                loss = self.loss(output, target)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update the progress bar
                iterator.set_postfix(loss=loss.item())
                #iterator.set_description(f"Train loss: {loss.detach().cpu().numpy()}")
                
                # Get the predictions
                preds.append(output)
                targets.append(target)
            
            preds = torch.cat(preds, axis=0)
            targets = torch.cat(targets, axis=0)

            # Calculate the loss and accuracy for the training set
            train_loss = self.loss(preds, targets)
            train_acc = accuracy_score(targets.detach().cpu().numpy(), preds.detach().cpu().numpy().argmax(axis=1))

            self.model.eval()
            with torch.no_grad():
                preds, targets = [], []
                for batch in self.test_loader:
                    # Get the data
                    if self.channels==1:
                        data = batch[0].unsqueeze(1).to(self.device) #### batch['data'].to(device)
                    else:
                        data = batch[0].to(self.device)
                    target = batch[1].to(self.device)

                    # Forward pass
                    output = self.model(data)
                    loss = self.loss(output, target)

                    # Get the predictions
                    preds.append(output)
                    targets.append(target)
                preds = torch.cat(preds, axis=0)
                targets = torch.cat(targets, axis=0)

                # Calculate the loss and accuracy for the test set
                test_loss = self.loss(preds, targets)
                test_acc = accuracy_score(targets.detach().cpu().numpy(), preds.detach().cpu().numpy().argmax(axis=1))


                print(f'Test loss: {test_loss.detach().cpu().numpy():.2f}')
                print(f'Test accuracy: {test_acc:.2f}')

                # Save the loss and accuracy values for plotting later
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                test_losses.append(test_loss)
                test_accs.append(test_acc)
                
            # Add the loss to Tensorboard
            #self.writer.add_scalar('Loss/train', loss, epoch)


            # Save the model
            if checkpoint:
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, checkpoint_path))

        # don't like this, to be changed
        train_losses = [x.detach().cpu().numpy() for x in train_losses]

        # Create dataframe with the losses and accuracies history
        self.history = pd.DataFrame({
            'train_loss': train_losses,
            'train_acc': train_accs,
            'test_loss': test_losses,
            'test_acc': test_accs
        })

        self.model_trained = True


    ######################################
    ######           CHECK         #######
    ######################################
    def evaluate_model(self,
                       do_cm: bool=True,
                       do_acc: bool=True,
                       do_prec_rec_f1: bool=True,
                       do_roc_auc: bool=True,
                       save: bool=True,
                       ):
        """
        Evaluate the model.

        Confusion matrix, accuracy, precision, recall, f1-score,
        ROC curve, AUC, etc.

        Parameters
        ----------
        do_cm : bool, optional
            Plot the confusion matrix. The default is True.
        do_acc : bool, optional
            Print the accuracy. The default is True.
        do_prec_rec_f1 : bool, optional
            Print the precision, recall and f1-score. The default is True.
        do_roc_auc : bool, optional
            Plot the ROC curve and print the AUC. The default is True.
        save : bool, optional
            Save the plots. The default is True.
            

        Raises
        ------
        WorkToDoError
            If the model is not trained.
        """
        if not self.model_trained:
            raise WorkToDoError('model_trained')

        # Set the model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            # Get the predictions
            preds, targets = [], []
            for batch in self.test_loader:
                # Get the data
                if self.channels==1:
                    data = batch[0].unsqueeze(1).to(self.device) #### batch['data'].to(device)
                else:
                    data = batch[0].to(self.device)
                target = batch[1]

                # Forward pass
                output = self.model(data)
                preds.append(output)
                targets.append(target)
            preds = torch.cat(preds, axis=0)
            targets = torch.cat(targets, axis=0)
            loss = self.loss(preds, targets).detach().cpu().numpy()
            preds = preds.detach().cpu().numpy().argmax(axis=1)
            targets = targets.detach().cpu().numpy()

        # Confusion matrix
        if do_cm:
            self.confusion_matrix(targets=targets, preds=preds, save=save)

        # Accuracy
        if do_acc:
            self.accuracy(targets=targets, preds=preds, save=save)

        # Precision, recall, f1-score
        if do_prec_rec_f1:
            self.precision_recall_fscore_support(targets=targets, preds=preds, save=save)

        # ROC curve
        if do_roc_auc:
            self.roc_auc_curve(targets=targets, preds=preds, save=save)


    def confusion_matrix(self, targets, preds, save: bool=True, show: bool=False):
        """
        Calculate the confusion matrix

        Parameters
        ----------
        targets : numpy.ndarray
            The targets.
        preds : numpy.ndarray
            The predictions.
        save : bool, optional
            Save the plot. The default is False.
        """

        # Calculate the confusion matrix
        cm = confusion_matrix(targets, preds)
        print(cm)

        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_title('Confusion matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

        if show:
            plt.show()

        # Save the confusion matrix
        if save:
            fig.savefig(os.path.join(fig_dir, f'{self.model_type}__confusion_matrix.png'))


    def accuracy(self, targets, preds, save: bool=False):
        """
        Calculate the accuracy
 --weight_decay=0. --no-nesterov
            The targets.
        preds : numpy.ndarray
            The predictions.
        save : bool, optional
            Save the accuracy. The default is False.
        """
        preds = preds.round()
        accuracy = accuracy_score(targets, preds)
        print(f'Accuracy: {accuracy}')

        # Save the accuracy
        if save:
            with open(os.path.join(fig_dir, f'{self.model_type}__accuracy.txt'), 'w') as f:
                f.write(f'Accuracy: {accuracy}')


    def precision_recall_fscore_support(self,
                                        targets,
                                        preds,
                                        average: str=None,
                                        save: bool=False):
        """
        Calculate the precision, recall and f1-score

        Parameters
        ----------
        targets : numpy.ndarray
            The targets.
        preds : numpy.ndarray
            The predictions.
        average : str, optional
            The averaging strategy. The default is None.
            Possible values are: None, 'micro', 'macro', 'weighted'.
        save : bool, optional
            Save the precision, recall and f1-score. The default is False.
        """
        preds = preds.round()
        precision, recall, fscore, _= precision_recall_fscore_support(targets, preds, average=None)

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-score: {fscore}')

        # Save the precision, recall, f1-score
        if save:
            with open(os.path.join(fig_dir, f'{self.model_type}__precision_recall_fscore_support.txt'), 'w') as f:
                f.write(f'Precision: {precision}\n')
                f.write(f'Recall: {recall}\n')
                f.write(f'F1-score: {fscore}\n')


    def _roc_curve(self,
                   y_pred_prob:np.ndarray,
                   y_true:np.ndarray,
                   return_roc:bool=True,
                   display:bool=True,
                   save:bool=True,
                   dir:str=None,
                   name:str=None,
                   **kwargs):
        """
        Compute the ROC curve of the model.

        Parameters
        ----------
        y_pred_prob: np.array
            Predicted label probabilities.
            Default: None
        y_true: np.array
            True labels.
            Default: None
        return_roc: bool
            Whether to return the ROC curve or not.
            Default: True
        display: bool
            Whether to display the figure or not.
            Default: True
        save: bool
            Whether to save the figure or not.
            Default: True
        dir: str
            Directory to save the figure.
            Default: None
        name: str
            Name of the figure.
            Default: None
        **kwargs:
            Parameters for the ROC curve method.
        """
        if not self.trained:
            raise ValueError('Model not trained')

        # Binarize the labels
        y_binarized = label_binarize(y_true, classes=range(self.num_classes))

        # Compute the ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and AUC
        fpr_micro, tpr_micro, _ = roc_curve(y_binarized.ravel(), y_pred_prob.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        if display or save:
            # Plot the ROC curves for each class
            plt.figure(figsize=(10, 10))
            for i in range(self.num_classes):
                plt.plot(fpr[i], tpr[i], label='Class {0} (AUC = {1:.2f})'.format(i, roc_auc[i]))
            plt.plot(fpr_micro, tpr_micro, label='Micro-average (AUC = {0:.2f})'.format(roc_auc_micro))

            plt.plot([0, 1], [0, 1], 'k--')  # Plot diagonal line
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - {0}'.format(self.model_type))
            #plt.legend(loc='lower right')
            
            if display:
                plt.show()
            
            if save:
                if dir is None:
                    dir='figures'
                if name is None:
                    name='roc_curve.png'
                plt.savefig(
                    os.path.join(
                        dir,
                        name
                    )
                )
                
        if return_roc:
            return fpr, tpr, roc_auc


    def _pr_curve(self,
                  y_pred_prob:np.ndarray,
                  y_true:np.ndarray,
                  return_pr:bool=True,
                  display:bool=True,
                  save:bool=True,
                  dir:str=None,
                  name:str=None,
                  **kwargs):
        """
        Compute the PR curve of the model.

        Parameters
        ----------
        y_pred_prob: np.array
            Predicted label probabilities.
            Default: None
        y_true: np.array
            True labels.
            Default: None
        return_pr: bool
            Whether to return the PR curve or not.
            Default: True
        display: bool
            Whether to display the figure or not.
            Default: True
        save: bool
            Whether to save the figure or not.
            Default: True
        dir: str
            Directory to save the figure.
            Default: None
        name: str
            Name of the figure.
            Default: None
        **kwargs:
            Parameters for the PR curve method.
            
        Returns
        -------
        precision : np.ndarray
            Precision values.
        """
        if not self.trained:
            raise ValueError('Model not trained')

        # Binarize the labels
        y_binarized = label_binarize(y_true, classes=range(self.num_classes))
        
        # Compute the PR curve and AUC for each class
        precision = dict()
        recall = dict()
        pr_auc = dict()
        for i in range(self.num_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_binarized[:, i], y_pred_prob[:, i])
            pr_auc[i] = auc(recall[i], precision[i])
        
        # Compute micro-average PR curve and AUC
        precision_micro, recall_micro, _ = precision_recall_curve(y_binarized.ravel(), y_pred_prob.ravel())
        pr_auc_micro = auc(recall_micro, precision_micro)
        
        if display or save:
            fig, ax = plt.subplots(figsize=(10, 10))

            for i in range(self.num_classes):
                ax.plot(recall[i], precision[i], label='Class {0} (AUC = {1:.2f})'.format(i, pr_auc[i]))
            ax.plot(recall_micro, precision_micro, label='Micro-average (AUC = {0:.2f})'.format(pr_auc_micro))

            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('PR Curve - {0}'.format(self.model_type))
            #ax.legend(loc='lower right')
            
            if display:
                plt.show()
            
            if save:
                if dir is None:
                    dir='figures'
                if name is None:
                    name='pr_curve.png'
                plt.savefig(
                    os.path.join(
                        dir,
                        name
                    )
                )
                
        if return_pr:
            return precision, recall, pr_auc


    def predict(self, data):
        """
        Predict the output

        Parameters
        ----------
        data : numpy.ndarray
            The data to predict.

        Returns
        -------
        numpy.ndarray
            The predictions.

        Raises
        ------
        WorkToDoError
            If the model is not trained.
        """
        if not self.model_trained:
            raise WorkToDoError('model_trained')

        # Convert the data to a tensor
        data = torch.tensor(data, dtype=torch.float32)

        # Predict the output
        with torch.no_grad():
            preds = self.model(data)
            preds = torch.sigmoid(preds).detach().cpu().numpy()

        return preds


    def plot_history(self,
                     save: bool=False,
                     path: str='figures',
                     name: str='history.png',
                     show: bool=True,
                     save_csv: bool=False,
                     path_csv: str='results',
                     name_csv: str='history.csv'
                     ):
        """
        Plot the history of the training

        Parameters
        ----------
        save : bool, optional
            Save the plot. The default is False.
        path : str, optional
            The path to save the plot. The default is 'figures'.
        name : str, optional
            The name of the plot. The default is 'history.png'.
        show : bool, optional
            Show the plot. The default is True.
        save_csv : bool, optional
            Save the history as a csv file. The default is False.
        path_csv : str, optional
            The path to save the csv file. The default is 'results'.
        name_csv : str, optional
            The name of the csv file. The default is 'history.csv'.

        Returns
        -------
        pandas.DataFrame
            The history of the training.
        """
        plotting.PlotHistory(self.history).plot(save=save, path=path, name=name, show=show)

        if save_csv:
            self.history.to_csv(os.path.join(path_csv, name_csv), index=False)

        return self.history


    def summary(self, save: bool=False, path: str='log', name: str='summary.txt'):
        """
        Print the summary of the model
        """
        if not self.train_test_split_done:
            raise WorkToDoError('train_test_split_done')
        if not self.model_created:
            raise WorkToDoError('model_created')
        
        if self.channels == 1:
            summary(self.model, (1,)+self.input_size)
        else:
            summary(self.model, self.input_size)
        
        if save:
            # Redirect stdout to a file
            sys.stdout = open(os.path.join(path, name), 'w')

            # Generate summary
            if self.channels == 1:
                summary(self.model, (1,)+self.input_size)
            else:
                summary(self.model, self.input_size)

            # Reset stdout
            sys.stdout = sys.__stdout__


    def save_trained_model(self, name: str, path: str='trained_models'):
        """
        Save the model

        Parameters
        ----------
        name : str
            The name of the model.
        path : str, optional
            The path to save the model. The default is 'trained_models'.

        Raises
        ------
        WorkToDoError
            If the model is not trained.
        """
        if not self.model_trained:
            raise WorkToDoError('model_trained')

        # Create the path if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Save the model
        torch.save(self.model.state_dict(), os.path.join(path, f'{name}.pt'))


    def load_model(self, name: str, path: str='trained_models'):
        """
        Load the model

        Parameters
        ----------
        name : str
            The name of the model.
        path : str, optional
            The path to load the model. The default is 'trained_models'.
        """
        # Load the model
        self.model.load_state_dict(torch.load(os.path.join(path, f'{name}.pt')))

        # Set the model as trained
        self.model_trained = True
