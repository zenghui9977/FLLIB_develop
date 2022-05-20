from fllib.base import *

config = {
    'dataset': {
        'data_name': 'mnist',
        'download': False,
        'distribution_type': 'non_iid_dir',
        'alpha': 0.5,
        'simulated': True
    },
    'server': {
        'rounds': 20,
        'clients_per_round': 10,
        'aggregation_rule': 'feddyn',
        'aggregation_detail': {
            'f': 2,
            'm': 0.3,
            'rho': 0.0005,
            'b': 1,
            'mu': 0.001,
            'feddyn_alpha': 0.001
        },
        'model_name': 'LeNet5'
    },
    'client': {
        'local_epoch': 2,
        'batch_size': 50,
        'optimizer':{
            'type': 'SGD',
            'lr': 0.01
            
        }
    },
    'trial_name': 'test',
    'resume': True

}

init(config=config)

run()
