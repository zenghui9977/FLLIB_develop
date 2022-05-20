
import logging
import copy
import time
import numpy as np
import torch
from fllib.client.base import BaseClient
import torchmetrics

logger = logging.getLogger(__name__)


PRE_MODEL = 'pre_model'
LOCAL_GRAD_VECTOR = 'local_grad_vector'

NABLA_L = 'nabla_l'
ALPHA = 'alpha'
CLIENT_ACC = 'train_acc'
CLIENT_LOSS = 'train_loss'

max_norm = 10

def reshape_model_param_torch(model):
    param_reshape = None
    for w in model.parameters():
        if not isinstance(param_reshape, torch.Tensor):
            param_reshape = w.reshape(-1)
        else:
            param_reshape = torch.cat((param_reshape, w.reshape(-1)), 0)
    return param_reshape


class FedDynClient(BaseClient):
    '''
    Paper:
    (2021 ICLR) Federated Learning Based on Dynamic Regularization 
    url: https://openreview.net/pdf?id=B7v4QMR6Z9w
    '''
    def __init__(self, config, device):
        super(FedDynClient, self).__init__(config, device)
        


    def download(self, model, **kwargs):

        if self.local_model is not None:
            self.local_model.load_state_dict(model.state_dict())
        else:
            self.local_model = copy.deepcopy(model)

        kwargs = kwargs['kwargs']
        if PRE_MODEL in kwargs.keys():
            self.pre_model = kwargs[PRE_MODEL].to(self.device)
        if LOCAL_GRAD_VECTOR in kwargs.keys():
            self.local_grad_vector = kwargs[LOCAL_GRAD_VECTOR].to(self.device) 
        if NABLA_L in kwargs.keys():
            self.nabla_l = kwargs[NABLA_L].to(self.device)
        if ALPHA in kwargs.keys():
            self.alpha = kwargs[ALPHA]
                

    def train(self, client_id, local_trainset):
        '''
        The local training process of FedDyn
        '''
        start_time = time.time()
        loss_fn, optimizer = self.train_preparation()

        train_accuracy = torchmetrics.Accuracy().to(self.device)

        # pre_global_model = reshape_model_param_torch(self.local_model).to(self.device)


        for e in range(self.config.local_epoch):
            batch_loss = []
            train_accuracy.reset()

            for imgs, labels in local_trainset:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs = self.local_model(imgs)

                theta = reshape_model_param_torch(self.local_model).to(self.device)

                # feddyn_term = self.alpha * torch.sum(theta * (-self.avg_model_params + self.local_grad_vector))
                feddyn_term = - torch.sum(theta * self.nabla_l) + (self.alpha / 2) * torch.sum((theta - self.pre_model) * (theta - self.pre_model))

                loss = loss_fn(outputs, labels) + feddyn_term

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
