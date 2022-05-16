import copy
import numpy as np
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


def compute_distance(v1, v2):
    distance = []
    for key in v1.keys():
        distance.append(torch.linalg.norm(v1[key].float() - v2[key].float()))
    return np.linalg.norm(distance, ord==2)


def get_closests(w, f):
    client_num = len(w)
    closests = np.zeros((client_num, client_num))
    for i in range(client_num):
        for j in range(client_num):
            if i < j:
                distance = compute_distance(w[i], w[j])
                closests[i][j] = distance
                closests[j][i] = distance
    if 2 * f + 2 > client_num:
        f = int(np.floor((client_num - 2)/2.0))
    thr = client_num - 2 - f
    closests = np.sort(closests)[:, 1:(1+thr)]

    return closests

def get_krum_scores(closests):
    scores = closests.sum(axis=-1)
    return scores

def Krum(local_updates, f, m):
    '''Aggregate the local updates by using Krum

    Paper:
    (2017 NIPS) Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent
    url: https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf

    Args:
        local_updates: the local updates including the data size from the selected clients
        f: the krum threshold of the closests /neighbors
        m: the number of the local updates 
    '''
    local_models = [local_updates[i]['model'] for i in local_updates.keys()]
    closests = get_closests(w=local_models, f=f)

    scores = get_krum_scores(closests)

    m = int(len(local_models) * m)
    min_idx = scores.argsort()[:m]
    aggregated_model_dict = mimic_blank_model(local_models[0])
    # aggregate the selected m local updates, the distance of these updates are more closer than others
    
    # multi-Krum, if Krum, set the m = 1
    with torch.no_grad():
        for name, param in aggregated_model_dict.items():
            for i in min_idx:
                param = param + torch.div(local_models[i][name], m)
            aggregated_model_dict[name] = param
    
    return aggregated_model_dict


def Zeno(local_updates):
    '''Aggregate the local updates by using Zeno
    
    
    Args:
        local_updates: the local updates including the data size from the selected clients
    '''
    #TODO
    pass

    
                





