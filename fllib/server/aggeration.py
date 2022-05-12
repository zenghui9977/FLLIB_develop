import copy

import torch


def mimic_blank_model(model_proto):
    blank_model_dict = dict()
    for name, params in model_proto.items():
        blank_model_dict[name] = torch.zeros_like(params)
    return blank_model_dict


def FedAvg(local_updates, agg_type='equal'):
    '''Aggregate the local updates by using FedAvg

    Args:
        local_updates: the local updates including the data size from the selected clients.
        type: aggregation type in FedAvg, options: equal, weight_by_size
            equal: all the local updates(model parameters directly average)
            weight_by_size: the local updates with a weight, which depends on the local data size.
    
    Return:
        aggregated model
    '''
    local_models = [local_updates[i]['model'] for i in local_updates.keys()]
    local_datasize = [local_updates[i]['size'] for i in local_updates.keys()]
    weights = [i/sum(local_datasize) for i in local_datasize]

    updates_num = len(local_updates)
    aggregated_model_dict = mimic_blank_model(local_models[0])
    with torch.no_grad():
        for name, param in aggregated_model_dict.items():
            if agg_type == 'equal':
                for i in range(updates_num):
                    param = param + torch.div(local_models[i][name], updates_num)
                aggregated_model_dict[name] = param
            elif agg_type == 'weight_by_size':
                for i in range(updates_num):
                    param = param + torch.mul(local_models[i][name], weights[i])
                aggregated_model_dict[name] = param

    return aggregated_model_dict




    
                





