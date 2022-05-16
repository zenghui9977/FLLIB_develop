from fllib.base import *

config = {
    'dataset': {
        'data_name': 'mnist',
        'download': False,
        'distribution_type': 'non_iid_dir',
        'alpha': 0.5
    },
    'server': {
        'rounds': 20,
        'clients_per_round': 10,
        'aggregation_rule': 'krum',
        'aggregation_detail': {
            'f': 2,
            'm': 0.3
        },
        'model_name': 'LeNet5'
    },
    'client': {
        'local_epoch': 2,
        'batch_size': 64,
        'optimizer':{
            'type': 'Adam'
        }
    },
    'trial_name': 'test'

}

init(config=config)

run()
