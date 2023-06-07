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
from utils.torch               import EarlyStopping
from torch.utils.data          import DataLoader
from torch.optim               import SGD, Adam, NAdam
from torch.nn                  import CrossEntropyLoss
from preprocessing.dataset     import Dataset
from exceptions                import OptionIsFalseError, WorkToDoError
from models.cnn_rd             import cnn_rd
from models.cnn_md             import cnn_md
from models.cnn_md_baseline    import cnn_md_baseline
from models.ResNet             import ResNet50
from models.inception_v4       import InceptionV4
from tqdm                      import tqdm
from sklearn.metrics           import confusion_matrix, accuracy_score,\
                                      precision_recall_fscore_support,\
                                      roc_curve, roc_auc_score,\
                                      precision_recall_curve, average_precision_score,\
                                      classification_report, auc, precision_score,\
                                      recall_score, f1_score
from utils                     import plotting
from torchsummary              import summary
from memory_profiler           import profile
from sklearn.preprocessing     import label_binarize
from sklearn.utils             import class_weight
from utils.constants           import NON_AGGREGATED_LABELS_DICT_REVERSE,\
    AGGREGATED_LABELS_DICT_REVERSE, MAPPING_LABELS_DICT
#from torch.utils.tensorboaro import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR,\
    CosineAnnealingLR, MultiStepLR, ExponentialLR

fig_dir = 'figures'
sns.set_style('darkgrid')

class model(object):
    """
    Class to create the model
    """
    model_created = False
    optimizer_created = False
    loss_created = False
    model_trained = False
    scheduler_created = False

    def __init__(
        self, 
        train_data: Dataset,
        test_data: Dataset,
        case: int=0,
        model_type: str='CNN-MD',
        device= torch.device('cuda' if torch.cuda.is_available() else 'cpu'),    
    ):
        """
        Constructor

        Parameters
        ----------
        train_data : Dataset
            Dataset object containing the training data
        test_data : Dataset
            Dataset object containing the test data
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
        self.train_data = train_data
        self.test_data = test_data
        
        self.input_size = self.train_data[0][0].shape

        # Get the case
        self.case = case

        # Get the model type
        self.model_type = model_type
        
        # Get the device
        self.device = device
        
        # Add tensorboard writer
        #self.writer = SummaryWriter()


    def create_DataLoaders(self,
                           batch_size: int=32,
                           shuffle: bool=True,
                           num_workers: int=os.cpu_count()):
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


    def create_model(self,
                     num_classes:int=10,
                     **kwargs):
        """
        Create the model
        
        Parameters
        ----------
        **kwargs : TYPE
            Keyword arguments to pass to the model class.
        """
        self.num_classes = num_classes

        # call cnn_rd or cnn_md class
        if self.model_type == 'CNN-MD':
            if self.train_data.TYPE != 'mDoppler':
                OptionIsFalseError('do_mDoppler')
            self.model = cnn_md(
                out_channels=num_classes,
                **kwargs
            )
        elif self.model_type == 'CNN-RD':
            if self.test_data.type != 'rdn':
                OptionIsFalseError('do_rdn')
            self.model = cnn_rd(
                out_channels=num_classes,
                **kwargs
            )
        elif self.model_type == 'CNN-MD-Baseline':
            if self.train_data.TYPE != 'mDoppler':
                OptionIsFalseError('do_mDoppler')
            self.model = cnn_md_baseline(
                out_channels=num_classes,
                **kwargs
            )
        elif self.model_type == 'ResNet':
            if self.train_data.TYPE != 'mDoppler':
                OptionIsFalseError('do_mDoppler')
            self.model = ResNet50(
                num_classes=num_classes,
                **kwargs
            )
        elif self.model_type == 'InceptionV4':
            if self.train_data.TYPE != 'mDoppler':
                OptionIsFalseError('do_mDoppler')
            self.model = InceptionV4(
                num_classes=num_classes,
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
            self.optimizer = SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum
            )
        elif optimizer == 'Adam':
            if not nesterov:
                self.optimizer = Adam(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
            else:
                self.optimizer = NAdam(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
        else:
            raise ValueError('Invalid optimizer')

        # Set the flag
        self.optimizer_created = True


    def create_loss(self,
                    loss: str='CrossEntropyLoss',
                    use_weight: bool=False,
                    weight: list=None,
                    **kwargs):
        """
        Create the loss function

        Parameters
        ----------
        loss : str, optional
            Loss function to use. The default is 'CrossEntropyLoss'.
            Possible values are:
                'CrossEntropyLoss': Cross Entropy Loss
        use_weight : bool, optional
            Use weight for the loss function. The default is False.
        weight : list, optional
            Weight for the loss function. The default is None.
            If None, the weight is set to the proportion of each class.
        **kwargs : TYPE
            Keyword arguments to pass to the loss function.

        Raises
        ------
        ValueError
            Invalid loss function.
        """
        
        if use_weight:
            if weight is None:
                if self.num_classes == 10:
                    labels_transform = np.vectorize(
                        lambda label: MAPPING_LABELS_DICT[label]
                    )
                    labels = labels_transform(self.train_data.labels[:])
                else:
                    labels = self.train_data.dataset.labels[:]
                class_weights = torch.tensor(
                    class_weight.compute_class_weight(
                        class_weight='balanced',
                        classes=np.unique(labels),
                        y=labels
                    )
                )
            else:
                class_weights = torch.tensor(weight)
        else:
            class_weights = None
        
        if loss == 'CrossEntropyLoss':
            self.loss = CrossEntropyLoss(
                weight=class_weights,
                **kwargs
            )
        #elif loss == :
            #self.loss = BCEWithLogitsLoss()
        else:
            raise ValueError('Invalid loss function')

        self.loss.weight = self.loss.weight.float()

        # Set the flag
        self.loss_created = True


    def create_early_stopping(
        self,
        patience: int=10,
        min_delta: float=0.0,
        verbose: bool=False,
        mode: str='min',
        baseline: float=None,
        start_epoch: int=0,
        path: str='chekpoints/checkpoint.pt',
    ):
        """
        Create the early stopping object
        
        Parameters
        ----------
        patience : int, optional
            Number of epochs with no
            improvement after which training will be stopped.
            The default is 10.
        delta : float, optional
            Minimum change in the monitored
            The default is 0.0.
        verbose : bool, optional
            Verbosity mode. The default is False.
        baseline : float, optional
            Baseline value for the monitored quantity.
            Training will stop if the model doesn't
            show improvement over the baseline.
            The default is None.
        start_epoch : int, optional
            Epoch from which to start counting the patience.
            The default is 0.
        path : str, optional
            Path for the checkpoint. The default is 'checkpoint.pt'.
        """
        self.early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            verbose=verbose,
            mode=mode,
            path=path,
            baseline=baseline,
            start_from_epoch=start_epoch
        )


    def create_scheduler(
        self,
        scheduler: str='ReduceLROnPlateau',
        **kwargs
    ):
        """
        Create the scheduler object
        
        Parameters
        ----------
        scheduler : str, optional
            Scheduler to use. The default is 'ReduceLROnPlateau'.
            Possible values are:
                'ReduceLROnPlateau': Reduce learning rate when a
                    metric has stopped improving.
                'StepLR': Decay the learning rate of each parameter
                    group by gamma every step_size epochs.
                'MultiStepLR': Decays the learning rate of each
                    parameter group by gamma once the number of
                    epoch reaches one of the milestones.
                'ExponentialLR': Decays the learning rate of each
                    parameter group by gamma every epoch.
                'CosineAnnealingLR': Set the learning rate of each
                    parameter group using a cosine annealing 
                    schedule.
        **kwargs : TYPE
            Keyword arguments to pass to the scheduler.
        """

        if not self.optimizer_created:
            raise WorkToDoError('Create the optimizer first')

        if scheduler == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                **kwargs
            )
        elif scheduler == 'StepLR':
            self.scheduler = StepLR(
                self.optimizer,
                **kwargs
            )
        elif scheduler == 'MultiStepLR':
            self.scheduler = MultiStepLR(
                self.optimizer,
                **kwargs
            )
        elif scheduler == 'ExponentialLR':
            self.scheduler = ExponentialLR(
                self.optimizer,
                **kwargs
            )
        elif scheduler == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                **kwargs
            )
        else:
            raise ValueError('Invalid scheduler')

        # Set the flag
        self.scheduler_created = True


    #@profile
    def train_model(self,
                    epochs: int=10,
                    accumulate_grad: int=1,
                    ):
        """
        Train the model

        Parameters
        ----------
        epochs : int, optional
            Number of epochs. The default is 10.
        checkpoint : bool, optional
            Save the model after every epoch. The default is False.

        Raises
        ------
        WorkToDoError
            If the train_test_split, create_model, create_optimizer,
            create_loss methods have not been called
        """
        if not self.model_created:
            raise WorkToDoError('model_created')
        if not self.optimizer_created:
            raise WorkToDoError('optimizer_created')
        if not self.loss_created:
            raise WorkToDoError('loss_created')

        # Use GPU
        self.model = self.model.to(self.device)

        # Define first best loss
        #best_loss = np.inf

        # Lists to store the losses and accuracies
        #train_losses = []
        #train_accs = []
        test_losses = []
        test_accs = []
        try:
            # Train the model
            for epoch in range(epochs):
                self.model.train()
                print(f'Epoch {epoch+1}/{epochs}')
                iterator = tqdm(self.train_loader)
                for batch_features, batch_targets in iterator:
                    # Get the data
                    data = batch_features.to(self.device)
                    target = batch_targets.long().to(self.device)

                    # Forward pass
                    output = self.model(data)
                    loss = self.loss(output, target)

                    # Backward pass
                    loss.backward()
    
                    # gradient accumulation to avoid memory overflow                
                    if ((iterator.n + 1) % accumulate_grad == 0) or (iterator.n == len(iterator) - 1):
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # Update the progress bar
                    iterator.set_postfix(loss=loss.item())

                # Calculate the loss and accuracy for the training set
                #train_loss = self.loss(preds, targets)
                #train_acc = accuracy_score(targets.detach().cpu().numpy(), preds.detach().cpu().numpy().argmax(axis=1))

                self.model.eval()
                with torch.no_grad():
                    preds, targets = [], []
                    iterator = tqdm(self.test_loader)
                    for batch_features, batch_targets in iterator:
                        # Get the data
                        data = batch_features.to(self.device)
                        target = batch_targets.long().to(self.device)

                        # Forward pass
                        output = self.model(data)

                        # Get the predictions
                        preds.append(output)
                        targets.append(target)
                    preds = torch.cat(preds, axis=0)
                    targets = torch.cat(targets, axis=0)
                    test_loss = self.loss(output, target)
                    test_acc = accuracy_score(targets.detach().cpu().numpy(), preds.detach().cpu().numpy().argmax(axis=1))

                    print(f'Test loss: {test_loss.detach().cpu().numpy():.2f}')
                    print(f'Test accuracy: {test_acc:.2f}')

                    # Save the loss and accuracy values for plotting later
                    #train_losses.append(train_loss)
                    #train_accs.append(train_acc)
                    test_losses.append(test_loss)
                    test_accs.append(test_acc)
                    
                # Add the loss to Tensorboard
                #self.writer.add_scalar('Loss/train', loss, epoch)

                # Save the model if the loss is the best we've seen so far
                self.early_stopping.check_improvement(
                    test_loss,
                )

                if self.scheduler_created:
                    self.scheduler.step(
                        metrics=test_loss,
                    )
        except KeyboardInterrupt:
            print('Interrupted by the user')
            print('Removing extra elements from the history')
            #train_losses = train_losses[:epoch]
            #train_accs = train_accs[:epoch]
            test_losses = test_losses[:epoch]
            test_accs = test_accs[:epoch]
            print('Last epoch: ', epoch)
            print('Best loss: ', self.early_stopping.best_score)


        # don't like this, to be changed
        #train_losses = [x.detach().cpu().numpy() for x in train_losses]

        # Create dataframe with the losses and accuracies history
        self.history = pd.DataFrame({
            #'train_loss': train_losses,
            #'train_acc': train_accs,
            'test_loss': test_losses,
            'test_acc': test_accs
        })

        self.model_trained = True


    def evaluate_model(self,
                       do_cm: bool=True,
                       do_acc: bool=True,
                       do_prec_rec_f1: bool=True,
                       do_roc_auc: bool=True,
                       do_pr_curve: bool=True,
                       do_classification_report: bool=True,
                       do_f1_score: bool=True,
                       do_recall_score: bool=True,
                       do_precision_score: bool=False,
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
            for batch_features, batch_targets in self.test_loader:
                # Get the data
                data = batch_features.to(self.device)
                target = batch_targets.long().to(self.device)

                # Forward pass
                output = self.model(data)
                preds.append(output)
                targets.append(target)
            preds = torch.cat(preds, axis=0)
            targets = torch.cat(targets, axis=0)
            loss = self.loss(preds, targets).detach().cpu().numpy()
            probs = preds.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy().argmax(axis=1)
            targets = targets.detach().cpu().numpy()

        if self.num_classes==10:
            target_names = AGGREGATED_LABELS_DICT_REVERSE.values()
        else:
            target_names = NON_AGGREGATED_LABELS_DICT_REVERSE.values()
            

        # Confusion matrix
        if do_cm:
            self.confusion_matrix(
                targets=targets,
                preds=preds,
                target_names=target_names,
                save=save
            )

        # Accuracy
        if do_acc:
            self.accuracy(targets=targets, preds=preds, save=save)

        # Precision, recall, f1-score
        if do_prec_rec_f1:
            self.precision_recall_fscore_support(targets=targets, preds=preds, save=save)

        # ROC curve
        if do_roc_auc:
            self._roc_curve(
                y_pred_prob=probs,
                y_true=targets,
                target_names=target_names,
                save=save,
                return_roc=False,
                display=False,
                dir=fig_dir,
            )
            
        # PR curve
        if do_pr_curve:
            self._pr_curve(
                y_true=targets,
                y_pred_prob=probs,
                target_names=target_names,
                save=save,
                return_pr=False,
                display=False,
                dir=fig_dir,
            )
            
        # Classification report
        if do_classification_report:
            print(
                self._classification_report(
                    y_true=targets,
                    y_pred=preds,
                    target_names=target_names,
                )
            )
            

    def _precision(self,
                   y_true:np.ndarray,
                   y_pred:np.ndarray,
                   average:str='macro',
                   **kwargs):
        """
        Compute the precision.
        
        Parameters
        ----------
        y_true : np.ndarray
            The true labels.
        y_pred : np.ndarray
            The predicted labels.
        average : str
            The averaging method.
        **kwargs : dict
            The arguments to pass to the precision object.
            
        Returns
        -------
        precision : float
            The precision.
        """
        return precision_score(
            y_true,
            y_pred,
            average=average,
            **kwargs
        )


    def _recall(self,
                y_true:np.ndarray,
                y_pred:np.ndarray,
                average:str='macro',
                **kwargs):
        """
        Compute the recall.
        
        Parameters
        ----------
        y_true : np.ndarray
            The true labels.
        y_pred : np.ndarray
            The predicted labels.
        average : str
            The averaging method.
        **kwargs : dict
            The arguments to pass to the recall object.
            
        Returns
        -------
        recall : float
            The recall.
        """
        return recall_score(
            y_true,
            y_pred,
            average=average,
            **kwargs
        )


    def _f1(self,
            y_true:np.ndarray,
            y_pred:np.ndarray,
            average:str='macro',
            **kwargs):
        """
        Compute the F1 score.
        
        Parameters
        y_true : np.ndarray
            The true labels.
        y_pred : np.ndarray
            The predicted labels.
        **kwargs : dict
            The arguments to pass to the F1 score object.
            
        Returns
        -------
        f1 : float
            The F1 score.
        """
        return f1_score(
            y_true,
            y_pred,
            average=average,
            **kwargs
        )
        
        
    def _classification_report(self,
                               y_true:np.ndarray,
                               y_pred:np.ndarray,
                               target_names:list=None,
                               **kwargs):
        """
        Compute the classification report.
        
        Parameters
        ----------
        y_true : np.ndarray
            The true labels.
        y_pred : np.ndarray
            The predicted labels.
        target_names : list
            The names of the classes.
        **kwargs : dict
            The arguments to pass to the classification report object.
            
        Returns
        -------
        classification_report : str
            The classification report.
        """
        
        return classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            **kwargs
        )


    def _roc_curve(self,
                   y_pred_prob:np.ndarray,
                   y_true:np.ndarray,
                   target_names:list=None,
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
        if not self.model_trained:
            raise ValueError('Model not trained')

        target_names = list(target_names)

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
                plt.plot(fpr[i], tpr[i], label='{0} (AUC = {1:.2f})'.format(target_names[i], roc_auc[i]))
            plt.plot(fpr_micro, tpr_micro, label='Micro-average (AUC = {0:.2f})'.format(roc_auc_micro))

            plt.plot([0, 1], [0, 1], 'k--')  # Plot diagonal line
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {self.model_type}')
            plt.legend(loc='lower right', )            
            
            if display:
                plt.show()
            
            if save:
                if dir is None:
                    dir='figures'
                if name is None:
                    name=f'{self.model_type}_roc_curve.png'
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
                  target_names:list=None,
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
        target_names: list
            The names of the classes.
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
        if not self.model_trained:
            raise ValueError('Model not trained')

        target_names = list(target_names)

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
                ax.plot(recall[i], precision[i], label='{0} (AUC = {1:.2f})'.format(target_names[i], pr_auc[i]))
            ax.plot(recall_micro, precision_micro, label='Micro-average (AUC = {0:.2f})'.format(pr_auc_micro))

            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'PR Curve - {self.model_type}')
            ax.legend(loc='upper right', )
            
            if display:
                plt.show()
            
            if save:
                if dir is None:
                    dir='figures'
                if name is None:
                    name=f'{self.model_type}_pr_curve.png'
                plt.savefig(
                    os.path.join(
                        dir,
                        name
                    )
                )
                
        if return_pr:
            return precision, recall, pr_auc


    def confusion_matrix(self,
                         targets,
                         preds,
                         target_names: list=None,
                         save: bool=True,
                         show: bool=False):
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
        cm = confusion_matrix(
            targets,
            preds
        )
        print(cm)

        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            ax=ax,
            xticklabels=target_names,
            yticklabels=target_names
        )
        ax.set_title('Confusion matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        fig.tight_layout()

        if show:
            plt.show()

        # Save the confusion matrix
        if save:
            fig.savefig(os.path.join(fig_dir, f'{self.model_type}__confusion_matrix.png'))


    def accuracy(self, targets, preds, save: bool=False):
        """
        Calculate the accuracy

        Parameters
        ----------
        targets : numpy.ndarray
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
        precision, recall, fscore, _= precision_recall_fscore_support(targets, preds, average=average)

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-score: {fscore}')

        # Save the precision, recall, f1-score
        if save:
            with open(os.path.join(fig_dir, f'{self.model_type}__precision_recall_fscore_support.txt'), 'w') as f:
                f.write(f'Precision: {precision}\n')
                f.write(f'Recall: {recall}\n')
                f.write(f'F1-score: {fscore}\n')



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
                     save_csv: bool=True,
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


    def summary(self,
                save: bool=False,
                path: str='log',
                name: str='summary.txt'):
        """
        Print the summary of the model
        
        Parameters
        ----------
        save : bool, optional
            Save the summary. The default is False.
        path : str, optional
            The path to save the summary. The default is 'log'.
        name : str, optional
            The name of the summary. The default is 'summary.txt'.
        """
        if not self.model_created:
            raise WorkToDoError('model_created')
        
        print(self.input_size)
        summary(self.model, self.input_size)
        
        if save:
            # Redirect stdout to a file
            sys.stdout = open(os.path.join(path, name), 'w')

            # Generate summary
            summary(self.model, self.input_size)

            # Reset stdout
            sys.stdout = sys.__stdout__


    def save_trained_model(self,
                           name: str,
                           path: str='trained_models'):
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


    def load_model(self,
                   name: str,
                   path: str='trained_models'):
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