import copy
import csv
import time
import numpy as np
import torch
import logging
import torchmetrics
import os
import gc
from fllib.server.aggeration import FedAvg, Krum, Marginal_Median, Zeno
from fllib.server.visualization import vis_scalar
import itertools

GLOBAL_ROUND = 'Round'
GLOBAL_ACC = 'Accuracy'
GLOBAL_LOSS = 'Loss'

logger = logging.getLogger(__name__)
class BaseServer(object):
    def __init__(self, config, clients, client_class, global_model, fl_trainset, testset, device, current_round=0, records_save_filename=None, vis=None):
        self.config = config.server
        
        self.client_class = client_class
        self.clients = clients
        self.selected_clients = None

        self.global_model = copy.deepcopy(global_model)
        self.aggregated_model_dict = None
      
        self.train_time = 0
        self.fl_trainset = fl_trainset
        self.testset = testset
        self.train_batchsize = config.client.batch_size 
    
        self.local_updates= {}

        self.device = device
        
        self.current_round = current_round

        self.train_records = {GLOBAL_ROUND: [], GLOBAL_ACC: [], GLOBAL_LOSS: []}
        self.records_save_filename = records_save_filename
        
        if not config.resume:
            self.write_one_row(one_raw=list(self.train_records.keys()), save_path=self.config.records_save_folder, save_file_name=self.records_save_filename)

        self.vis = vis

        if self.config.aggregation_rule == 'zeno':
            zeno_dataloader = copy.deepcopy(self.testset)
            self.zeno_iter = itertools.cycle(zeno_dataloader)


    def one_step(self):
        '''One round training process in the server
        '''
        logger.info('----- Round {}th -----'.format(self.current_round))
        start_time = time.time()
        # clients selection
        self.client_selection(clients=self.clients, clients_per_round=self.config.clients_per_round)
        self.client_training()
        self.aggregation()
        self.update_global_model()
        loss, acc = self.test()


        self.train_records[GLOBAL_ROUND] = self.current_round 
        self.train_records[GLOBAL_ACC] = acc
        self.train_records[GLOBAL_LOSS] = loss
        
        self.write_one_row(one_raw=[self.current_round, acc, loss], save_path=self.config.records_save_folder, save_file_name=self.records_save_filename)
        

        self.train_time = time.time() - start_time
        logger.debug('{}th round use {:.4f}s.'.format(self.current_round, self.train_time))
        if self.vis is not None:
            vis_scalar(vis=self.vis, figure_name=f'{self.config.records_save_folder}/{self.records_save_filename}/{GLOBAL_ACC}', scalar_name=GLOBAL_ACC, x=self.current_round, y=acc)
            vis_scalar(vis=self.vis, figure_name=f'{self.config.records_save_folder}/{self.records_save_filename}/{GLOBAL_LOSS}', scalar_name=GLOBAL_LOSS, x=self.current_round, y=loss)
            
        self.current_round += 1
        
        return self.global_model

    def multiple_steps(self):
        for _ in range(self.config.rounds):
            self.one_step()
            self.save_the_checkpoint(save_path=self.config.records_save_folder, save_file_name=self.records_save_filename + '_checkpoint')
            
            gc.collect()
            torch.cuda.empty_cache()     
        return self.global_model


    def client_selection(self, clients, clients_per_round):
        '''Select the client in one round.
        
        Args:
            clients: list[Object:'BaseClient']. 
            clients_id: list[int], index or clients_id list
            clients_per_round: int;  the number of clients in each round        
        
        Return:
            selected_clients: list[Object:'BaseClient'] with length clients_per_round
        '''
        if clients_per_round >= len(clients):
            logger.warning('Clients for selected are smaller than the required.')
        
        clients_per_round = min(len(clients), clients_per_round)
        if self.config.random_select:
            self.selected_clients = np.random.choice(clients, clients_per_round, replace=False)
        else:
            self.selected_clients = clients[:clients_per_round]
        
        return self.selected_clients

        
    def client_training(self):
        '''The global model is distributed to these selected clients.
        And the clients start local training.
        '''
        if len(self.selected_clients) > 0:
            for client in self.selected_clients:
                self.local_updates[client] = {
                    'model': self.client_class.step(global_model=self.global_model, 
                                                    client_id=client, 
                                                    local_trainset=self.fl_trainset.get_dataloader(client, batch_size=self.train_batchsize)).state_dict(),
                    'size': self.fl_trainset.get_client_datasize(client_id=client)
                }
        
        else:
            logger.warning('No clients in this round')
            self.local_updates = None
        return self.local_updates
        

    def aggregation(self):
        '''Different aggregation methods

        Return:
            The aggregated global model
        '''
        
        aggregation_algorithm = self.config.aggregation_rule
        if self.local_updates is None:
            self.aggregated_model_dict = self.global_model.state_dict()
            
        else:
            if aggregation_algorithm in ['fedavg', 'fedprox', 'scaffold', 'feddyn']:
                self.aggregated_model_dict = FedAvg(self.local_updates, agg_type=self.config.aggregation_detail.type)
            elif aggregation_algorithm == 'krum':
                self.aggregated_model_dict = Krum(self.local_updates, f=self.config.aggregation_detail.f, m=self.config.aggregation_detail.m)
            elif aggregation_algorithm == 'zeno':
                loss_fn = self.load_loss_function()
                samples = self.zeno_iter.__next__()
                self.aggregated_model_dict = Zeno(local_updates=self.local_updates, 
                                                pre_global_model=self.global_model, 
                                                loss_fn=loss_fn,
                                                samples=samples,
                                                rho=self.config.aggregation_detail.rho,
                                                b=self.config.aggregation_detail.b)
            # median is not effective
            # elif aggregation_algorithm == 'median':
            #     self.aggregated_model_dict = Marginal_Median(local_updates=self.local_updates)


        return self.aggregated_model_dict

    def update_global_model(self):
        if self.global_model is not None:
            self.global_model.load_state_dict(self.aggregated_model_dict)
        return self.global_model
        
    
    def test(self):
        '''Test the current global model
        '''
        
        self.global_model.eval()
        self.global_model.to(self.device)

        logger.debug('Test in the server')

        loss_fn = self.load_loss_function()

        test_accuracy = torchmetrics.Accuracy().to(self.device)

        with torch.no_grad():
            batch_loss = []
            for imgs, labels in self.testset:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs = self.global_model(imgs)
                batch_loss.append(loss_fn(outputs, labels).item())

                _ = test_accuracy(outputs, labels)
            
            total_test_loss = np.mean(batch_loss)
            total_test_accuracy = test_accuracy.compute().item()

            logger.info('Loss: {:.4f}, Accuracy: {:.4f}'.format(total_test_loss, total_test_accuracy))

            return total_test_loss, total_test_accuracy
        
    def load_loss_function(self):
        if self.config.loss_fn == 'cross_entropy':
            return torch.nn.CrossEntropyLoss()
        elif self.config.loss_fn == 'mse':
            return torch.nn.MSELoss()
        else: 
            # defualt is cross entropy
            return torch.nn.CrossEntropyLoss().to(self.device)

    def set_global_model(self, model):
        self.global_model = copy.deepcopy(model)

    def save_global_model(self, save_path, save_file_name):
        if os.path.exists(os.path.join(save_path, save_file_name)):
            logger.info('Overwrite the existing file {}'.format(os.path.join(save_path, save_file_name)))
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        
        torch.save(self.global_model, os.path.join(save_path, save_file_name))


    def save_the_checkpoint(self, save_path, save_file_name):
        if os.path.exists(os.path.join(save_path, save_file_name)):
            logger.info('Overwrite the existing file {}'.format(os.path.join(save_path, save_file_name)))
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        
        checkpoint = {
            'model': self.global_model,
            'round': self.current_round
        }

        torch.save(checkpoint, os.path.join(save_path, save_file_name))
        

    def save_test_records(self, save_path, save_file_name):
        header = list(self.train_records.keys())
        content = [[r, a, l] for r,a,l in zip(self.train_records[GLOBAL_ROUND], self.train_records[GLOBAL_ACC], self.train_records[GLOBAL_LOSS])]

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        with open(os.path.join(save_path, save_file_name)+'.csv', 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(content)

    def write_one_row(self, one_raw, save_path, save_file_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(os.path.join(save_path, save_file_name) + '.csv', 'a', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(one_raw)
    
    



