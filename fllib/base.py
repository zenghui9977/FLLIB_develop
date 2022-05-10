import datetime
import os
import sys
import time
import logging
import copy
import random
from fllib.datasets.base import BaseDataset, FederatedDataset
from fllib.client.base import BaseClient
from fllib.server.base import BaseServer
from fllib.models.base import load_model
from visdom import Visdom
import numpy as np
import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

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
        self.server = None
        self.clients = None
        self.config = None
        self.clients_id = None
        
        self.source_dataset = None
        self.testset_loader = None
        self.fl_dataset = None

        self.vis = None
        
        self.global_model = None
        self.exp_name = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    def init_config(self, config):
        '''Initialize the configuration from yaml files or dict from the code

        Args:
            config: input configurations: dict or yaml file
        
        Return:
            configurations
        '''
        self.config = config

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

        self.testset_loader = self.fl_dataset.get_dataloader(client_id=None, 
                                                            batch_size=self.config.client.test_batch_size,
                                                            istrain=False)

        self.exp_name =  datetime.datetime.now().strftime('%Y%m%d%H%M') + self.fl_dataset.store_file_name

        logger.info('FL dataset distributed successfully.')

        return self.fl_dataset


    def init_server(self):
        logger.info('Server initialization.')
        self.server = BaseServer(config=self.config, 
                                clients=self.clients, 
                                global_model=self.global_model, 
                                testset=self.testset_loader, 
                                device=self.device,
                                records_save_filename=self.exp_name,
                                vis=self.vis
                                )
     
        

    def init_clients_id(self):
        self.clients_id = []
        for c in range(self.config.server.clients_num):
            self.clients_id.append("clients%05.0f" % c)
        return self.clients_id


    def init_clients(self):
        logger.info('Clients initialization.')
        self.clients = []
        for cid in self.clients_id:
            local_trainset_loader = self.fl_dataset.get_dataloader(cid, batch_size=self.config.client.batch_size)
            local_testset_loader = self.fl_dataset.get_dataloader(cid, batch_size=self.config.client.batch_size, istrain=False)
            train_datasize = self.fl_dataset.get_client_datasize(client_id=cid)
            client = BaseClient(client_id=cid, 
                                config=self.config, 
                                local_trainset=local_trainset_loader, 
                                local_testset=local_testset_loader, 
                                device=self.device, 
                                train_datasize=train_datasize)

            self.clients.append(client)

        
        return self.clients


    def init_global_model(self, global_model=None):
        if global_model is not None:
            self.global_model = copy.deepcopy(global_model)
        else:
            self.global_model = load_model(self.config.server.model_name)

        return self.global_model

    
    def init_visualization(self, vis=None):
        if vis is not None:
            self.vis = vis
        else:
            self.vis = Visdom()


        
    def init_fl(self, config=None, global_model=None, fl_dataset=None):
        self.init_config(config=config)
        self.init_global_model(global_model=global_model)

        self.init_fl_dataset(fl_dataset=fl_dataset)
        self.init_clients()

        if self.config.is_visualization:
            self.init_visualization()

        self.init_server()

    def run(self):
        start_time = time.time()

        self.server.multiple_steps()
        logger.info('Total training time {:.4f}s'.format(time.time() - start_time))


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


