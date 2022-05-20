import datetime
import os
import sys
import time
import logging
import copy
import random

import yaml
from fllib.client.feddyn import FedDynClient
from fllib.client.fedprox import FedProxClient
from fllib.client.scaffold import ScaffoldClient
from fllib.datasets.base import BaseDataset, FederatedDataset
from fllib.client.base import BaseClient
from fllib.server.feddyn import FedDynServer
from fllib.server.scaffold import ScaffoldServer
from fllib.server.base import BaseServer
from fllib.models.base import load_model
from visdom import Visdom
import numpy as np
import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


support_optimizer = ['Adam', 'SGD']


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class BaseFL(object):
    ''' BasedFL class coordinates the server and clients in FL.
    Each time the package is imported, a instance of BaseFL will be initilized.
    '''
    def __init__(self):
        self.config = None
        self.server = None
        
        self.client_class = None    
        self.clients_id = None
        
        self.source_dataset = None
        self.testset_loader = None
        self.fl_dataset = None

        self.vis = None
        
        self.global_model = None
        self.model_channel = 1
        self.num_class = 10

        self.exp_name = None
        

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    def init_config(self, config, exp_name=None):
        '''Initialize the configuration from yaml files or dict from the code

        Args:
            config: input configurations: dict or yaml file
        
        Return:
            configurations
        '''
        self.config = config

        self.init_exp_name(exp_name=exp_name)

        # save the param file
        OmegaConf.save(config=self.config, f=f'{self.config.server.records_save_folder}/param.yaml')
        logger.info('Configurations loaded.')

    def init_source_dataset(self, source_dataset=None):
        if source_dataset is not None:
            self.source_dataset = source_dataset
        else:
            self.source_dataset = BaseDataset(datatype=self.config.dataset.data_name, 
                                            root=self.config.dataset.root, 
                                            download=self.config.dataset.download)
        
        logger.debug('Source data loaded.')
        return self.source_dataset.get_dataset()    
        
    def init_fl_dataset(self, fl_dataset=None):
        if fl_dataset is not None:
            self.fl_dataset = fl_dataset
        else:
            trainset, testset = self.init_source_dataset()

            self.model_channels = trainset[0][0].shape[0]
            
            self.clients_id = self.init_clients_id()
            self.fl_dataset = FederatedDataset(data_name=self.config.dataset.data_name,
                                                trainset=trainset,
                                                testset=testset,
                                                simulated=self.config.dataset.simulated,
                                                simulated_root=self.config.dataset.simulated_root,
                                                distribution_type=self.config.dataset.distribution_type,
                                                clients_id=self.clients_id,
                                                class_per_client=self.config.dataset.class_per_client,
                                                alpha=self.config.dataset.alpha,
                                                min_size=self.config.dataset.min_size
                                                )

            self.num_class = self.fl_dataset.get_num_class()


        self.testset_loader = self.fl_dataset.get_dataloader(client_id=None, 
                                                            batch_size=self.config.client.test_batch_size,
                                                            istrain=False)

        # self.exp_name =  datetime.datetime.now().strftime('%Y%m%d%H%M') + self.fl_dataset.store_file_name

        logger.info('FL dataset distributed successfully.')

        return self.fl_dataset


    def init_server(self, current_round=0):
        if self.config.server.aggregation_rule  == 'scaffold':
            logger.info('Scaffold Server initialization.')
            self.server = ScaffoldServer(config=self.config, 
                                    clients=self.clients_id, 
                                    client_class=self.client_class,
                                    global_model=self.global_model,
                                    fl_trainset=self.fl_dataset,
                                    testset=self.testset_loader, 
                                    device=self.device,
                                    current_round=current_round,
                                    records_save_filename=self.config.trial_name,
                                    vis=self.vis)

        elif self.config.server.aggregation_rule  == 'feddyn':
            logger.info('FedDyn Server initialization.')
            self.server = FedDynServer(config=self.config, 
                                    clients=self.clients_id, 
                                    client_class=self.client_class,
                                    global_model=self.global_model,
                                    fl_trainset=self.fl_dataset,
                                    testset=self.testset_loader, 
                                    device=self.device,
                                    current_round=current_round,
                                    records_save_filename=self.config.trial_name,
                                    vis=self.vis)

        else:
            logger.info('Base Server initialization.')
            self.server = BaseServer(config=self.config, 
                                    clients=self.clients_id, 
                                    client_class=self.client_class,
                                    global_model=self.global_model,
                                    fl_trainset=self.fl_dataset,
                                    testset=self.testset_loader, 
                                    device=self.device,
                                    current_round=current_round,
                                    records_save_filename=self.config.trial_name,
                                    vis=self.vis)
        
            

    def init_clients_id(self):
        self.clients_id = []
        for c in range(self.config.server.clients_num):
            self.clients_id.append("clients%05.0f" % c)
        return self.clients_id


    def init_clients(self):

        if self.config.server.aggregation_rule == 'fedprox':
            logger.info('FedProx Clients initialization.')
            self.client_class = FedProxClient(config=self.config, device=self.device)
        elif self.config.server.aggregation_rule == 'scaffold':
            logger.info('Scaffold Clients initialization.')
            self.client_class = ScaffoldClient(config=self.config, device=self.device)
        elif self.config.server.aggregation_rule == 'feddyn':
            logger.info('FedDyn Clients initialization.')
            self.client_class = FedDynClient(config=self.config, device=self.device)
        else:
            logger.info('Base Clients initialization.')
            self.client_class = BaseClient(config=self.config, device=self.device)

      
        return self.client_class


    def init_global_model(self, global_model=None):
        if global_model is not None:
            self.global_model = copy.deepcopy(global_model)
        else:
            self.global_model = load_model(model_name=self.config.server.model_name, num_class=self.num_class, channels = self.model_channels)

        return self.global_model

    
    def init_visualization(self, vis=None):
        if vis is not None:
            self.vis = vis
        else:
            self.vis = Visdom()

    def init_exp_name(self, exp_name=None):
        if exp_name is not None:
            self.exp_name = exp_name
        else:
            # distribution_args
            if self.config.dataset.distribution_type == 'iid':
                distribution_args = 0
            elif self.config.dataset.distribution_type == 'non_iid_class':
                distribution_args = self.config.dataset.class_per_client
            elif self.config.dataset.distribution_type == 'non_iid_dir':
                distribution_args = self.config.dataset.alpha
            else:
                distribution_args = 0

            # optimizer_args

            if self.config.client.optimizer.type not in support_optimizer:
                optimizer_args = '{}_lr{}_m{}_de{}'.format(support_optimizer[0], self.config.client.optimizer.lr, self.config.client.optimizer.momentum, self.config.client.optimizer.weight_decay)
            else:
                optimizer_args = '{}_lr{}_m{}_de{}'.format(self.config.client.optimizer.type, self.config.client.optimizer.lr, self.config.client.optimizer.momentum, self.config.client.optimizer.weight_decay)

            
            # aggregation_detail
            if self.config.server.aggregation_rule == 'fedavg':
                aggregation_detail = '[{}]'.format(self.config.server.aggregation_detail.type) 
            elif self.config.server.aggregation_rule == 'krum':
                aggregation_detail = '[f{}m{}]'.format(self.config.server.aggregation_detail.f, self.config.server.aggregation_detail.m)
            elif self.config.server.aggregation_rule == 'zeno':
                aggregation_detail = '[rho{}b{}]'.format(self.config.server.aggregation_detail.rho, self.config.server.aggregation_detail.b)
            elif self.config.server.aggregation_rule == 'fedprox':
                aggregation_detail = '[mu{}]'.format(self.config.server.aggregation_detail.mu)
            elif self.config.server.aggregation_rule == 'scaffold':
                aggregation_detail = ''
            elif self.config.server.aggregation_rule == 'feddyn':
                aggregation_detail = '[alpha{}]'.format(self.config.server.aggregation_detail.feddyn_alpha)

            else:
                aggregation_detail = '[none]'

            
            
            self.exp_name = '{}_{}_N{}_p{}_{}_{}_E{}_{}'.format(self.config.dataset.data_name,
                                                        self.config.dataset.distribution_type + str(distribution_args),
                                                        self.config.server.clients_num,
                                                        self.config.server.clients_per_round,
                                                        self.config.server.aggregation_rule + aggregation_detail,
                                                        self.config.server.model_name,
                                                        self.config.client.local_epoch,
                                                        optimizer_args)

            
            self.config.server.records_save_folder = os.path.join(self.config.server.records_save_folder , self.exp_name)
            if not os.path.exists(self.config.server.records_save_folder):
                os.makedirs(self.config.server.records_save_folder)



        
    def init_fl(self, config=None, fl_dataset=None, global_model=None, server=None, client=None, exp_name=None, current_round=0):
        self.init_config(config=config, exp_name=exp_name)
        
        self.init_fl_dataset(fl_dataset=fl_dataset)

        if self.config.resume:
            if (global_model is None) and (current_round==0):
                global_model, current_round = self.load_checkpoint()
        
        self.init_global_model(global_model=global_model)

        self.init_clients()

        if self.config.is_visualization:
            self.init_visualization()

        self.init_server(current_round=current_round)

    def run(self):
        start_time = time.time()

        self.global_model = self.server.multiple_steps()
        logger.info('Total training time {:.4f}s'.format(time.time() - start_time))


    def load_checkpoint(self):
        if os.path.exists(f'{self.config.server.records_save_folder}/{self.config.trial_name}_checkpoint'):
            checkpoint = torch.load(f'{self.config.server.records_save_folder}/{self.config.trial_name}_checkpoint')
            global_model = checkpoint['model']
            current_round = checkpoint['round']
            logger.info(f'Checkpoint successfully loaded from {self.config.server.records_save_folder}/{self.config.trial_name}_checkpoint')
        else:
            global_model = None
            current_round = 0
        return global_model, current_round

global_fl = BaseFL()

def init_config(config=None):
    """Initialize configuration. 

    Args:
        conf (dict): Configurations.

    Returns:
        omegaconf.dictconfig.DictConfig: Internal configurations managed by OmegaConf.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    config_file = os.path.join(here, 'config_example.yaml')
    return merge_config(config_file, config)   

def merge_config(config_file, config_=None):
    """Load and merge configuration from file and input

    Args:
        file (str): filename of the configuration.
        conf (dict): Configurations.

    Returns:
        omegaconf.dictconfig.DictConfig: Internal configurations managed by OmegaConf.
    """
    config = OmegaConf.load(config_file)
    if config_ is not None:
        config = OmegaConf.merge(config, config_)
    return config

def init_logger(log_level):
    """Initialize internal logger of EasyFL.

    Args:
        log_level (int): Logger level, e.g., logging.INFO, logging.DEBUG
    """
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-6.5s]  %(message)s")
    root_logger = logging.getLogger()

    log_level = logging.INFO if not log_level else log_level
    root_logger.setLevel(log_level)

    file_path = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, "train" + time.strftime(".%m_%d_%H_%M_%S") + ".log")
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)    



def init(config=None):

    global global_fl
    
    config = init_config(config)

    init_logger(config.log_level)

    set_all_random_seed(config.seed)

    global_fl.init_fl(config)

def run():
    global global_fl

    global_fl.run()


