from fllib.base import *

config = {
    'server': {
        'rounds': 100
    },
    'dataset': {
        'download': False
    },
    'client': {
        'local_epoch': 1,
        'optimizer':{
            'type': 'SGD'
        }
    }

}

init(config=config)

run()
