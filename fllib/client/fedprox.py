import logging
import copy
import time
import numpy as np
import torch
from fllib.client.base import BaseClient
import torchmetrics

logger = logging.getLogger(__name__)

CLIENT_ACC = 'train_acc'
CLIENT_LOSS = 'train_loss'

max_norm = 10

class FedProxClient(BaseClient):
    
    def __init__(self, config, device):
        super(FedProxClient, self).__init__(config, device)
        self.mu = config.server.aggregation_detail.mu
       

    def train(self, client_id, local_trainset):
        ''' The local training process of FedProx
        '''
        start_time = time.time()
        loss_fn, optimizer = self.train_preparation()

        train_accuracy = torchmetrics.Accuracy().to(self.device)
        last_global_model = copy.deepcopy(self.local_model)

        for e in range(self.config.local_epoch):
            batch_loss = []
            train_accuracy.reset()
            for imgs, labels in local_trainset:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs = self.local_model(imgs)

                proximal_term = 0.0

                for w, w_t in zip(self.local_model.parameters(), last_global_model.parameters()):
                    proximal_term = proximal_term + (w - w_t).norm(2) 


                loss = loss_fn(outputs, labels) + (self.mu / 2) * proximal_term
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.local_model.parameters(), max_norm=max_norm)
                optimizer.step()

                batch_loss.append(loss.item())

                _ = train_accuracy(outputs, labels)

            current_epoch_loss = np.mean(batch_loss) 
            current_epoch_acc = train_accuracy.compute().item()

            self.train_records[CLIENT_LOSS].append(current_epoch_loss)
            self.train_records[CLIENT_ACC].append(current_epoch_acc)
            logger.debug('Client: {}, local epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(client_id, e, current_epoch_loss, current_epoch_acc))
        train_time = time.time() - start_time
        logger.debug('Client: {}, training {:.4f}s'.format(client_id, train_time))

