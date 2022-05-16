import logging
import os
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from fllib.datasets.simulation import data_distribution_simulation
from fllib.datasets.utils import TransformDataset


support_dataset = ['mnist', 'fmnist', 'kmnist', 'emnist', 'cifar10', 'cifar100']
logger = logging.getLogger(__name__)
class BaseDataset(object):
    '''The internal base dataset class, most of the dataset is based on the torch lib

    Args:
        type: The dataset name, options: mnist, fmnist, kmnist, emnist,cifar10
        root: The root directory of the dataset folder.
        download: The dataset should be download or not

    '''
    def __init__(self, datatype, root, download):
        self.type = datatype
        self.root = root
        self.download = download
        self.trainset = None
        self.testset = None
        self.idx_dict = {}
        
        # self.support_dataset = ['mnist', 'fmnist', 'kmnist', 'emnist', 'cifar10', 'cifar100']
        # self.get_dataset()
        

    def get_dataset(self):
        if self.type == 'mnist':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            self.trainset = torchvision.datasets.MNIST(root=self.root, train=True, transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.MNIST(root=self.root, train=False, transform=simple_transform, download=self.download)
        
        elif self.type == 'fmnist':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor(),])
            self.trainset = torchvision.datasets.FashionMNIST(root=self.root, train=True, transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.FashionMNIST(root=self.root, train=False, transform=simple_transform, download=self.download)
        
        elif self.type == 'kmnist':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            self.trainset = torchvision.datasets.KMNIST(root=self.root, train=True, transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.KMNIST(root=self.root, train=False, transform=simple_transform, download=self.download)

        elif self.type == 'emnist':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor()])
            self.trainset = torchvision.datasets.EMNIST(root=self.root, train=True, split='byclass', transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.EMNIST(root=self.root, train=False, split='byclass', transform=simple_transform, download=self.download)

        elif self.type == 'cifar10':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])
            self.trainset = torchvision.datasets.CIFAR10(root=self.root, train=True, transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.CIFAR10(root=self.root, train=False, transform=simple_transform, download=self.download)

        elif self.type == 'cifar100':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
            self.trainset = torchvision.datasets.CIFAR100(root=self.root, train=True, transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.CIFAR100(root=self.root, train=False, transform=simple_transform, download=self.download)

        else:
            raise ValueError(f'Dataset name is not correct, the options are listed as follows: {support_dataset}')

        return self.trainset, self.testset

        

class FederatedDataset(object):

    def __init__(self, data_name, trainset, testset, simulated, simulated_root, distribution_type, clients_id, class_per_client=2, alpha=0.9, min_size=1):
        
        self.trainset = trainset
        self.testset = testset
        self.idx_dict = self.build_idx_dict()

        
        
        self.data_name = data_name

        self.simulated = simulated
        self.simulated_root = simulated_root
        self.distribution_type = distribution_type

        self.clients_id = clients_id
        self.clients_num = len(clients_id)

        if self.distribution_type == 'iid':
            distribution_args = 0
        elif self.distribution_type == 'non_iid_class':
            distribution_args = class_per_client
        elif self.distribution_type == 'non_iid_dir':
            distribution_args = alpha

        self.store_file_name = f'{self.data_name}_{self.distribution_type}_clients{self.clients_num}_args{distribution_args}'

        if os.path.exists(os.path.join(self.simulated_root, self.store_file_name)) and (not self.simulated):
            logger.info(f'Clients data file {self.store_file_name} already exist. Loading......')
            self.clients_data = torch.load(os.path.join(simulated_root, self.store_file_name))
            
        else:
            if not os.path.exists(self.simulated_root):
                os.makedirs(self.simulated_root)
            logger.info(f'Initialize the file {self.store_file_name}.')
            self.clients_data = data_distribution_simulation(self.clients_id, self.idx_dict, self.distribution_type, class_per_client, alpha, min_size)
            torch.save(self.clients_data, os.path.join(self.simulated_root, self.store_file_name))

        

    def build_idx_dict(self):
        self.idx_dict = {}
        for idx, data in enumerate(self.trainset):
            _, label = data
            if label in self.idx_dict:
                self.idx_dict[label].append(idx)
            else:
                self.idx_dict[label] = [idx]
        return self.idx_dict

    def get_dataloader(self, client_id, batch_size, istrain=True, drop_last=False):
        if self.data_name in support_dataset:
            if istrain:
                if client_id in self.clients_id:
             
                    data_idx = self.clients_data[client_id]
                    imgs, labels = [], []

                    for i in data_idx:
                        imgs.append(self.trainset[i][0])
                        labels.append(self.trainset[i][1])
                
                    return DataLoader(dataset=TransformDataset(imgs, labels), batch_size=min(len(data_idx), batch_size), shuffle=True, drop_last=drop_last)

                else:
                    raise ValueError('The client id is not existed.')
            else:
                return DataLoader(dataset=self.testset, batch_size=batch_size, shuffle=True)
        
        else:
            raise ValueError(f'Dataset name is not correct, the options are listed as follows: {support_dataset}')
    
                
    def get_client_datasize(self, client_id=None):
        if client_id in self.clients_id:
            return len(self.clients_data[client_id])
        else:
            raise ValueError('The client id is not existed.')
    
    def get_total_datasize(self):
        return len(self.trainset)

    def get_num_class(self):
        return len(self.idx_dict)

    def get_client_datasize_list(self):
        return [len(self.clients_data[i]) for i in self.clients_id]
                
        