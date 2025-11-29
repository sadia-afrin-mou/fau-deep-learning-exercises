import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import os


def cleanup_checkpoints(checkpoint_dir, keep_best_n=3):
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith('checkpoint_') and f.endswith('.ckp'):
            try:
                epoch = int(f.split('_')[1].split('.')[0])
                checkpoints.append((epoch, os.path.join(checkpoint_dir, f)))
            except:
                continue
    
    if len(checkpoints) <= keep_best_n:
        return
    
    checkpoints.sort(reverse=True)
    for _, ckpt_path in checkpoints[keep_best_n:]:
        try:
            os.remove(ckpt_path)
            print(f"Removed checkpoint: {ckpt_path}")
        except:
            print(f"Failed to remove checkpoint: {ckpt_path}")


class Trainer:

    def __init__(self,
                 model,                        # model to be trained.
                 crit,                         # loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # training data set
                 val_test_dl=None,             # validation (or test) data set
                 cuda=True,                    # whether to use the gpu
                 early_stopping_patience=-1,   # the patience for early stopping
                 scheduler=None,               # optional lr scheduler
                 freeze_epochs=0):             # epochs to freeze pretrained backbone
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience
        self._scheduler = scheduler
        self._freeze_epochs = freeze_epochs

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        map_loc = 'cuda' if self._cuda else 'cpu'
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), map_location=map_loc)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        was_cuda = next(self._model.parameters()).is_cuda
        m = self._model.to('cpu')
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=False)
        t.onnx.export(m,
              x,
              fn,
              export_params=True,
              opset_version=10,
              do_constant_folding=True,
              input_names=['input'],
              output_names=['output'],
              dynamic_axes={'input' : {0 : 'batch_size'},
                            'output' : {0 : 'batch_size'}})
        if was_cuda:
            self._model = self._model.to('cuda')
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        
        # Reset gradients
        self._optim.zero_grad()
        
        # Forward propagation
        predictions = self._model(x)
        
        # Calculate loss
        loss = self._crit(predictions, y)
        
        # Backward propagation
        loss.backward()
        
        # Update weights
        self._optim.step()
        
        return loss.item()
        
    
    def val_test_step(self, x, y):
        
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        
        # Forward propagation
        predictions = self._model(x)
        
        # Calculate loss
        loss = self._crit(predictions, y)
        
        return loss.item(), predictions
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        
        self._model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for x, y in tqdm(self._train_dl, desc='Training'):
            # Transfer to GPU if available
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            
            # Perform training step
            batch_loss = self.train_step(x, y)
            epoch_loss += batch_loss
            num_batches += 1
        
        return epoch_loss / num_batches
    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        
        self._model.eval()
        epoch_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        with t.no_grad():
            for x, y in tqdm(self._val_test_dl, desc='Validation'):
                # Transfer to GPU if available
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                
                # Perform validation step
                batch_loss, predictions = self.val_test_step(x, y)
                epoch_loss += batch_loss
                num_batches += 1
                
                # Save predictions and labels for metrics calculation
                all_predictions.append(predictions.cpu())
                all_labels.append(y.cpu())
        
        # Concatenate all predictions and labels
        all_predictions = t.cat(all_predictions, dim=0)
        all_labels = t.cat(all_labels, dim=0)
        
        # Calculate metrics
        # Convert predictions to binary (threshold at 0.5)
        binary_predictions = (all_predictions > 0.5).float()
        
        # Calculate F1 scores for each class
        f1_crack = f1_score(all_labels[:, 0].numpy(), binary_predictions[:, 0].numpy(), average='binary')
        f1_inactive = f1_score(all_labels[:, 1].numpy(), binary_predictions[:, 1].numpy(), average='binary')
        
        avg_loss = epoch_loss / num_batches
        
        f1_mean = (f1_crack + f1_inactive) / 2
        print(f'Validation Loss: {avg_loss:.4f}, F1 Crack: {f1_crack:.4f}, F1 Inactive: {f1_inactive:.4f}, F1 Mean: {f1_mean:.4f}')
        
        return avg_loss
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        
        train_losses = []
        val_losses = []
        epoch_counter = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        while True:
            epoch_counter += 1
            
            # stop by epoch number
            if epochs > 0 and epoch_counter > epochs:
                break
            
            print(f'Epoch {epoch_counter}/{epochs if epochs > 0 else "∞"}')

            
            if self._freeze_epochs > 0 and hasattr(self._model, 'backbone'):
                is_freeze_phase = epoch_counter <= self._freeze_epochs
                trainable_params_before = sum(p.requires_grad for p in self._model.parameters())
                
                # during freeze phase: freeze all backbone layers except fc
                # after freeze phase: only unfreeze layer3, layer4 (and fc stays trainable)
                for name, p in self._model.backbone.named_parameters():
                    if 'fc' in name:
                        p.requires_grad = True 
                    elif not is_freeze_phase and ('layer3' in name or 'layer4' in name):
                        p.requires_grad = True  
                    else:
                        p.requires_grad = False 
                
                trainable_params_after = sum(p.requires_grad for p in self._model.parameters())
                
                # log freeze/unfreeze status
                if epoch_counter == 1:
                    if is_freeze_phase:
                        print(f"Backbone frozen for first {self._freeze_epochs} epochs. Trainable params: {trainable_params_after}")
                    else:
                        print(f"No freezing (freeze_epochs={self._freeze_epochs}). Trainable params: {trainable_params_after}")
                elif epoch_counter == self._freeze_epochs + 1:
                    print(f"Unfroze layer3 & layer4 at epoch {epoch_counter}. Trainable params: {trainable_params_before} -> {trainable_params_after}")
                
                
                if self._optim is not None:
                    trainable_params = [p for p in self._model.parameters() if p.requires_grad]
                    for g in self._optim.param_groups:
                        g['params'] = trainable_params
            
            # train for a epoch and then calculate the loss and metrics on the validation set.
            train_loss = self.train_epoch()
            val_loss = self.val_test()
            
            # append the losses to the respective lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch_counter)
                cleanup_checkpoints(os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"), keep_best_n=3)
                patience_counter = 0
                print(f'New best validation loss: {val_loss:.4f} - Model saved!')
            else:
                patience_counter += 1
            
            # Step scheduler if provided
            if self._scheduler is not None:
                if isinstance(self._scheduler, t.optim.lr_scheduler.ReduceLROnPlateau):
                    self._scheduler.step(val_loss)
                else:
                    self._scheduler.step()

            
            if self._early_stopping_patience > 0 and patience_counter >= self._early_stopping_patience:
                print(f'Early stopping triggered after {patience_counter} epochs without improvement')
                break
            
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print('-' * 50)
        
       
        return train_losses, val_losses
        
        
        
        
