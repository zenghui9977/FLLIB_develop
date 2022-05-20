import os
import copy
import logging
import numpy as np
import torch
from fllib.server.base import BaseServer
from fllib.server.aggeration import FedAvg


logger = logging.getLogger(__name__)

GLOBAL_HIST_NAME = 'global'
NABLA_LIST = 'nabla_list'


def reshape_model_param_torch(model):
    param_reshape = None
    for w in model.parameters():
        if not isinstance(param_reshape, torch.Tensor):
            param_reshape = w.reshape(-1)
        else:
            param_reshape = torch.cat((param_reshape, w.reshape(-1)), 0)
    return param_reshape

def reshape_model_param(model):
    param_reshape = np.asarray([])
    for w in model.parameters():
        param_reshape = np.hstack((param_reshape, w.detach().cpu().numpy().reshape(-1)))  
    return param_reshape

def reshape_model_param_dict(model_dict):
    param_reshape = np.asarray([])
    for k, w in model_dict.items():
        param_reshape = np.hstack((param_reshape, w.detach().cpu().numpy().reshape(-1)))  
    return param_reshape

def mimic_model_dict(param_list, model_dict, device):
    dict_param = copy.deepcopy(model_dict)
    idx = 0
    for k, w in model_dict.items():
        length = len(w.reshape(-1))
        dict_param[k] = torch.tensor(param_list[idx:idx+length].reshape(w.shape)).to(device)
        idx += length
    return dict_param




class FedDynServer(BaseServer):

    def __init__(self, config, clients, client_class, global_model, fl_trainset, testset, device, current_round=0, records_save_filename=None, vis=None):
        super(FedDynServer, self).__init__(config, clients, client_class, global_model, fl_trainset, testset, device, current_round, records_save_filename, vis)

        self.nabla_list = self.load_checkpoint_and_init_nabla_list(resume=config.resume, save_path=self.config.records_save_folder, save_file_name=self.records_save_filename+ '_checkpoint')
        self.weight_list = self.init_weight_list()
        self.alpha = self.config.aggregation_detail.feddyn_alpha
    
    
    def init_weight_list(self):
        weight_list = np.asarray(self.fl_trainset.get_client_datasize_list())
        weight_list = weight_list / np.sum(weight_list) * self.config.clients_num
        self.weight_list = {}
        for i in range(self.config.clients_num):
            self.weight_list[self.clients[i]] = weight_list[i]
        return self.weight_list

    
    def init_nabla_list(self):
        
        nabla_l = np.zeros_like(reshape_model_param(self.global_model)).astype('float32')
        self.nabla_list = {}
        for i in range(self.config.clients_num):
            self.nabla_list[self.clients[i]] = nabla_l
        
        return self.nabla_list

    def client_training(self):

        pre_model = reshape_model_param(self.global_model)
        
        if len(self.selected_clients) > 0:
            
            for client in self.selected_clients:
                nabla_l = torch.tensor(self.nabla_list[client], dtype=torch.float32, device=self.device)
                local_update = self.client_class.step(global_model=self.global_model, 
                                                    client_id=client, 
                                                    local_trainset=self.fl_trainset.get_dataloader(client, batch_size=self.train_batchsize),
                                                    pre_model=reshape_model_param_torch(self.global_model),
                                                    nabla_l=nabla_l,
                                                    alpha=self.alpha/self.weight_list[client])
                
                # dtype: numpy
                # self.nabla_list[client] += reshape_model_param(local_update) - pre_model
                self.nabla_list[client] = self.nabla_list[client] - self.alpha/self.weight_list[client] * (reshape_model_param(local_update) - pre_model)


                self.local_updates[client] = {
                    'model': local_update.state_dict(),
                    'size': self.fl_trainset.get_client_datasize(client_id=client)
                }
        
        else:
            logger.warning('No clients in this round')
            self.local_updates = None
        return super().client_training()


    def aggregation(self):
        if self.local_updates is None:
            self.aggregated_model_dict = self.global_model.state_dict()
        else:

            avg_dict = FedAvg(self.local_updates)
            nabla_list_mean = np.mean(list(self.nabla_list.values()), axis=0)
           
            model_param = reshape_model_param_dict(avg_dict) - 1/self.alpha * nabla_list_mean
            
            self.aggregated_model_dict = mimic_model_dict(model_param, avg_dict, device=self.device)

        return self.aggregated_model_dict


    def save_the_checkpoint(self, save_path, save_file_name):
        if os.path.exists(os.path.join(save_path, save_file_name)):
            logger.info('Overwrite the existing file {}'.format(os.path.join(save_path, save_file_name)))
        else: 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        
        checkpoint = {
            'model': self.global_model,
            'round': self.current_round,
            NABLA_LIST: self.nabla_list
        }
        
        torch.save(checkpoint, os.path.join(save_path, save_file_name))


    def load_checkpoint_and_init_nabla_list(self, resume, save_path, save_file_name):
        if resume and os.path.exists(os.path.join(save_path, save_file_name)):
            checkpoint = torch.load(f'{save_path}/{save_file_name}')       
            nabla_list = checkpoint[NABLA_LIST]
            logger.info('FedDyn: Loding the nabla list from the checkpoint')
        else:
            logger.info('FedDyn: Initialize control parameter')
            nabla_list = self.init_nabla_list()
        return nabla_list

            