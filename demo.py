from fllib.base import *

config = {
    'server': {
        'rounds': 1,
        'clients_per_round': 1
    },
    'dataset': {
        'download': False
    },
    'client': {
        'local_epoch': 1,
        'batch_size': 64,
        'optimizer':{
            'type': 'Scaffold'
        }
    },
    'trial_name': 'delete'

}

init(config=config)

run()
