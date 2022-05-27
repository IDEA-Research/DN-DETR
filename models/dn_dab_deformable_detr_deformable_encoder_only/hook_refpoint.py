import torch
import torch.nn as nn


@torch.no_grad()
def add_half_hw(query):
    """
    query:
        num_query, 4

    output:
        new_num_query, 4
    """
    query2 = query
    query2[:, 2:] /= 2
    return torch.cat((query, query2), dim=0)

@torch.no_grad()
def add_xy_plus_0p1(query):
    """
    query: num_query, 4
    output: 
    """
    query2 = query
    # query2[:, :2] += 0.1
    # query2 = torch.clip(query2, 0, 1)
    return torch.cat((query, query2), dim=0)

@torch.no_grad()
def offset_0p01(query):
    """
    query: num_query, 4
    output: 
    """
    query2 = query
    query2[:, :2] += 0.01
    query2 = torch.clip(query2, 0, 1)
    return query2