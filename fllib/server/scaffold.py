
import copy
import logging
import numpy as np
import torch
from fllib.server.base import BaseServer
import os


logger = logging.getLogger(__name__)

GLOBAL_CONTROL_PARAM_NAME = 'global'
CONTROL_PARAMETER = 'control_parameter'
CONTROAL_PARAMETER_DIFF = 'control_parameter_diff'

def reshape_model_param(model):
    param_reshape = np.asarray([])
    for w in model.parameters():
        param_reshape = np.hstack((param_reshape, w.detach().cpu().numpy().reshape(-1)))  
    return param_reshape

def local_update_diff_list(pre_model, cur_model):
    return reshape_model_param(pre_model) - reshape_model_param(cur_model)



class ScaffoldServer(BaseServer):
    def __init__(self, config, clients, client_class, global_model, fl_trainset, testset, device, current_round=0, records_save_filename=None, vis=None):
        super().__init__(config, clients, client_class, global_model, fl_trainset, testset, device, current_round, records_save_filename, vis)

        self.weight_list = self.init_weight_list()


        self.control_param = self.load_checkpoint_control_param(resume=config.resume, save_path=self.config.records_save_folder, save_file_name=self.records_save_filename+ '_checkpoint')

        self.n_minbatch = (np.ceil((self.fl_trainset.get_total_datasize() / self.config.clients_num)/self.train_batchsize) * config.client.local_epoch).astype(np.int64)

        
        self.lr = config.client.optimizer.lr

    def init_weight_list(self):
        weight_list = np.asarray(self.fl_trainset.get_client_datasize_list())
        weight_list = weight_list / np.sum(weight_list) * self.config.clients_num
        self.weight_list = {}
        for i in range(self.config.clients_num):
            self.weight_list[self.clients[i]] = weight_list[i]
        

        return self.weight_list

    def init_control_param(self):
        
        control_param = np.zeros_like(reshape_model_param(self.global_model)).astype('float32')
        self.control_param = {}
        for i in range(self.config.clients_num):
            self.control_param[self.clients[i]] = control_param
        
        self.control_param[GLOBAL_CONTROL_PARAM_NAME] = control_param
        return self.control_param


    def client_training(self):
        pre_model = copy.deepcopy(self.global_model)
        delta_control_param_sum = np.zeros_like(reshape_model_param(self.global_model))

        if len(self.selected_clients) > 0:
            for client in self.selected_clients:
                control_parameter_diff = - self.control_param[client] + self.control_param[GLOBAL_CONTROL_PARAM_NAME]/self.weight_list[client]
                
                size = self.fl_trainset.get_client_datasize(client_id=client)
                # local update is a model not state_dict()
                local_update = self.client_class.step(global_model=self.global_model, 
                                                    client_id=client, 
                                                    local_trainset=self.fl_trainset.get_dataloader(client, batch_size=self.train_batchsize),
                                                    control_parameter_diff=control_parameter_diff)
                
                new_control_param =  self.control_param[client] - self.control_param[GLOBAL_CONTROL_PARAM_NAME] + 1/self.n_minbatch/self.lr * local_update_diff_list(pre_model=pre_model, cur_model=local_update)
                
                delta_control_param_sum += (new_control_param  - self.control_param[client]) * self.weight_list[client]
                

                self.control_param[client] = new_control_param

                self.local_updates[client] = {
                    'model': local_update.state_dict(),
                    'size': size
                }

            self.control_param[GLOBAL_CONTROL_PARAM_NAME] += 1/len(self.clients) * delta_control_param_sum
            
        else:
            logger.warning('No clients in this round')
            self.local_updates = None
        return self.local_updates

    # rewrite the save_the_checkpoint and add load_checkpoint
    def save_the_checkpoint(self, save_path, save_file_name):
        if os.path.exists(os.path.join(save_path, save_file_name)):
            logger.info('Overwrite the existing file {}'.format(os.path.join(save_path, save_file_name)))
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        
        checkpoint = {
            'model': self.global_model,
            'round': self.current_round,
            CONTROL_PARAMETER: self.control_param
        }

        torch.save(checkpoint, os.path.join(save_path, save_file_name)) 


    def load_checkpoint_control_param(self, resume, save_path, save_file_name):
        if resume and os.path.exists(os.path.join(save_path, save_file_name)):
            checkpoint = torch.load(f'{save_path}/{save_file_name}')
            control_parameter = checkpoint[CONTROL_PARAMETER]
            logger.info('Scaffold: Loding the control parameter from the checkpoint')
        else:
            logger.info('Scaffold: Initialize control parameter')
            control_parameter = self.init_control_param()
        
        return control_parameter
