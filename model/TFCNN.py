from einops import repeat
from model.TSViT.TSViTdense import Transformer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from einops.layers.torch import Rearrange
from torch import optim
from matplotlib.patches import Rectangle
from torch.optim import lr_scheduler
from pathlib import Path
from utils.tools import update_confusion_matrix, on_test_epoch_end
from sklearn.metrics import roc_curve, auc
import os
class TFCNN(pl.LightningModule):
    def __init__(self, 
                dim, depth, heads, dim_head, mlp_dim,
                train_config = None
    ):
        super().__init__()
        self._init_train_params(train_config)

        self.dim = dim
        if self.dim == 12:
            self.classifier_dim = 64 * 7 * 2  
        elif self.dim == 10:
            self.classifier_dim = 64 * 6 * 2
        else:
            raise ValueError('Invalid dim value. Please use either 10 or 12')
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, dim))
        self.tem_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.to_temporal_embedding_input = nn.Linear(366, dim)


        self.feature_len = self.num_classes * dim
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature_len),
            # nn.Linear(self.feature_len, 1)
            nn.Linear(self.feature_len, self.num_classes)
        )
        # CNN architecture adjusted for input size (16, 12x12)
        self.features = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=2, padding=1),  # First layer adjusts for 16 input channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: (32, 6, 6)
            nn.Conv2d(32, 64, kernel_size=2, padding=1),  # Second convolution layer
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: (64, 3, 3)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(64 * 7 * 2, 128),  # for dim =12
            nn.Linear(self.classifier_dim, 128),  # Adjusted to match the output size of the last pooling layer
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
            # nn.Softmax(dim=1)
        )
        self.save_hyperparameters()

    def _init_train_params(self, train_config):
        self.epoch_train_losses = []
        self.epoch_valid_losses = []
        self.avg_train_losses = []
        self.avg_val_losses = []
        self.best_loss = None
        self.learning_rate = train_config['learning_rate']

        self.checkpoint_epoch = train_config['checkpoint_epoch']
        self.method = train_config['method']

        if 'class_weights' in train_config and train_config['class_weights'] is not None:
            class_weights_tensor = torch.tensor([train_config['class_weights'][k] for k in sorted(train_config['class_weights'].keys())]).cuda()

            self.lossfunction = nn.CrossEntropyLoss(
                weight=torch.tensor(list(train_config['class_weights'].values())).to('cuda')
                 )
        else:
            print('Class weights not found in train_config, using default NLLLoss')

        self.linear_encoder = train_config['linear_encoder']

        self.num_classes = len(set(train_config['linear_encoder'].values()))
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).to('cuda')

        self.run_path = Path(train_config['run_path'])

        self.pred_all = []
        self.label_all = []
    def _get_temporal_position_embeddings(self, t):
        t = (t * 365.0001).to(torch.int64)
        t = F.one_hot(t, num_classes=366).to(torch.float32)
        t = t.reshape(-1, 366)
        return self.to_temporal_embedding_input(t)
    def forward(self, x, time):
        
        if time.ndim == 3:
            time = time[:,0,:]

        temporal_pos_embedding = self._get_temporal_position_embeddings(time).unsqueeze(dim=0) # (b, time) ->(No. pixels, T, self.dim)

        x += temporal_pos_embedding

        x = self.tem_transformer(x) 
        # x = self.tem_transformer(x.squeeze(0)) 

        x = self.features(x[...,None])

        x = self.classifier(x)

        return x
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.pla_lr_scheduler = {
            'scheduler': lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        factor=0.5,
                                                        patience=15,
                                                        min_lr = 0.00008
                                                        ),
            'monitor': 'val_loss',
            'reduce_on_plateau': True,
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [self.pla_lr_scheduler]

    def _common_step(self, batch, batch_idx):
        inputs = batch['data'].float()  # (B, p, T, C, 1)
        time = batch['time']  # (1, T) or (1,pixels, T)    
        label = batch['label'].to(torch.long).squeeze()  # 0D tensor

        pred = self(inputs, time)  # (B, K (the number of categories), H, W)

        if label.dim() == 0 or label.size(0) == 1:
            label = torch.full((pred.size(0),), label, dtype=torch.long).to('cuda')

        loss = self.lossfunction(pred, label)
        return pred, loss, label, inputs
    def training_step(self, batch, batch_idx):
        pred, loss, label, inputs = self._common_step(batch, batch_idx)
        self.epoch_train_losses.append(loss.item())
        return {'loss': loss}
    def training_epoch_end(self, outputs):
        # Calculate average loss over an epoch
        train_loss = np.nanmean(self.epoch_train_losses)
        self.avg_train_losses.append(train_loss)

        with open(self.run_path / "avg_train_losses.txt", 'a') as f:
            f.write(f'{self.current_epoch}: {train_loss}\n')

        with open(self.run_path / 'lrs.txt', 'a') as f:
            f.write(f'{self.current_epoch}: {self.pla_lr_scheduler["scheduler"].get_last_lr()[0]}\n')

        self.log('train_loss', train_loss, prog_bar=True)

        # Clear list to track next epoch
        self.epoch_train_losses = []
    def validation_step(self, batch, batch_idx):
        pred, loss, label, inputs = self._common_step(batch, batch_idx)
        pred = pred.mean(dim=0, keepdim=True)
        label = label[0].reshape(1)
        val_loss = self.lossfunction(pred, label)

        self.epoch_valid_losses.append(val_loss.item())

        return {'val_loss': val_loss}
    def validation_epoch_end(self, outputs):
        # Calculate average loss over an epoch
        valid_loss = np.nanmean(self.epoch_valid_losses)
        self.avg_val_losses.append(valid_loss)

        with open(self.run_path / "avg_val_losses.txt", 'a') as f:
            f.write(f'{self.current_epoch}: {valid_loss}\n')

        self.log('val_loss', valid_loss, prog_bar=True)

        # Clear list to track next epoch
        self.epoch_valid_losses = []
    # def test_step(self, batch, batch_idx):
    #     pred, loss, label, inputs = self._common_step(batch, batch_idx)

    #     # for binary
    #     # pred = (pred > 0.5).float()
    #     pred = pred.mean(dim=0).argmax()
    #     label = label[0]

    #     self.confusion_matrix = update_confusion_matrix(self.confusion_matrix, label.view(-1), pred.view(-1))

    #     p = pred.view(-1).cpu().detach().numpy()
    #     l = label.view(-1).cpu().detach().numpy()
        
    #     p_binary = np.where(p < 7, 1, 0)
    #     l_binary = np.where(l < 7, 1, 0)

    #     self.pred_all.append(p_binary)
    #     self.label_all.append(l_binary)
    def test_step(self, batch, batch_idx):
        # Obtain predictions, loss, and labels from the common step function
        pred, loss, label, inputs = self._common_step(batch, batch_idx)

        # Do not convert pred into hard labels (keep them as probabilities)
        # pred = pred.mean(dim=0).argmax()  # Comment this line out

        # For each sample, obtain the probability for CTC (classes < 7) and Non-CTC (classes >= 7)
        p_probs = pred.mean(dim=0)  # Keep the soft predictions

        # Convert label (ground truth) into binary classification
        label = label[0]
        l_binary = np.where(label.view(-1).cpu().detach().numpy() < 7, 1, 0)  # 1 for CTC, 0 for Non-CTC

        # Compute probabilities for CTC: Summing probabilities of classes < 7 for binary ROC
        ctc_prob = np.array([p_probs[:7].sum().cpu().detach().numpy()])  # This gives the probability that the sample is CTC

        # Ensure both ctc_prob and l_binary are 1D arrays before appending
        self.pred_all.append(ctc_prob)  # Append probability of being CTC
        self.label_all.append(np.array([l_binary]))  # Append binary label (ground truth)

        # Update the confusion matrix with hard predictions (for other metrics)
        pred_hard = p_probs.argmax()  # Use hard label for confusion matrix
        self.confusion_matrix = update_confusion_matrix(self.confusion_matrix, label.view(-1), pred_hard.view(-1))
    def test_epoch_end(self, outputs):

        self.confusion_matrix = self.confusion_matrix.cpu().detach().numpy()

        # save_dir.mkdir(parents=True, exist_ok=True)
        
        on_test_epoch_end(self.run_path, self.checkpoint_epoch, self.confusion_matrix, self.linear_encoder)


            # Concatenate all predictions and labels collected during testing
        predictions = np.concatenate(self.pred_all)
        labels = np.concatenate(self.label_all)

        # Compute the ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)

        # Plot ROC Curve
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line for random classifier
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for CTC vs Non-CTC')
        plt.legend(loc="lower right")

        # Save ROC plot to the run_path
        roc_plot_path = os.path.join(self.run_path, f'roc_curve_epoch_{self.checkpoint_epoch}.png')
        plt.savefig(roc_plot_path)  # Save the plot to the specified directory
        plt.close() 
