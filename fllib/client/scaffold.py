import logging
import copy
import time
import numpy as np
import torch
from fllib.client.base import BaseClient
import torchmetrics

logger = logging.getLogger(__name__)

CONTROAL_PARAMETER_DIFF = 'control_parameter_diff'
CLIENT_ACC = 'train_acc'
CLIENT_LOSS = 'train_loss'

max_norm = 10


def mimic_model_dict(param_list, model_dict, device):
    dict_param = copy.deepcopy(model_dict)
    idx = 0
    for k, w in model_dict.items():
        length = len(w.reshape(-1))
        dict_param[k] = torch.tensor(param_list[idx:idx+length].reshape(w.shape)).to(device)
        idx += length
    return dict_param


class ScaffoldClient(BaseClient):
    '''
    Paper 
    (2020 ICML) SCAFFold: Stochastic Controlled Averaging for Federated Learning
    url: http://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf
    '''
    def __init__(self, config, device):
        super(ScaffoldClient, self).__init__(config, device)

    def train(self, client_id, local_trainset):
        '''
        The local training process of SCAFFOLD
        '''
        start_time = time.time()

        loss_fn, optimizer = self.train_preparation()

        train_accuracy = torchmetrics.Accuracy().to(self.device)

        for e in range(self.config.local_epoch):
            batch_loss = []
            train_accuracy.reset()

            for imgs, labels in local_trainset:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                

                outputs = self.local_model(imgs)

                # scaffold_term = 0.0
                # # scaffold term 
                # w = None
                # for param in self.local_model.parameters():
                #     if not isinstance(w, torch.Tensor):
                #     # Initially nothing to concatenate
                #         w = param.reshape(-1)
                #     else:
                #         w = torch.cat((w, param.reshape(-1)), 0)

                # scaffold_term = torch.sum(w * self.control_parameter_diff)
                
                loss = loss_fn(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.local_model.parameters(), max_norm=max_norm)
                optimizer.step()

                # update the local model using scaffold
                temp_local_model = self.local_model.state_dict()
                for k in temp_local_model:
                   temp_local_model[k] = temp_local_model[k] - self.config.optimizer.lr * self.control_parameter_diff[k]


                self.local_model.load_state_dict(temp_local_model)

                batch_loss.append(loss.item())

                _ = train_accuracy(outputs, labels)                           

            current_epoch_loss = np.mean(batch_loss) 
            current_epoch_acc = train_accuracy.compute().item()

            self.train_records[CLIENT_LOSS].append(current_epoch_loss)
            self.train_records[CLIENT_ACC].append(current_epoch_acc)
            logger.debug('Client: {}, local epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(client_id, e, current_epoch_loss, current_epoch_acc))
        train_time = time.time() - start_time
        logger.debug('Client: {}, training {:.4f}s'.format(client_id, train_time))


    def download(self, model, **kwargs):
        
        if self.local_model is not None:
            self.local_model.load_state_dict(model.state_dict())
        else:
            self.local_model = copy.deepcopy(model)

        kwargs = kwargs['kwargs']
        if CONTROAL_PARAMETER_DIFF in kwargs.keys():
            self.control_parameter_diff = kwargs[CONTROAL_PARAMETER_DIFF]
            self.control_parameter_diff = mimic_model_dict(self.control_parameter_diff, self.local_model.state_dict(), device=self.device)


