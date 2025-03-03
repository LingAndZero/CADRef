import torch
import numpy as np
from tqdm import tqdm
from utils.linear_mapping import get_linear_layer_mapping

class ASH:

    def __init__(self, model, args,device):
        self.model = model
        self.device = device
        self.linear = get_linear_layer_mapping(args.model, args.ind_dataset, model)

        '''
        Special Parameters:
            T--Temperature
            p--Pruning Percentage
        '''
        self.T = 1
        if args.ind_dataset in ['cifar10']:
            self.p = 95
        else:
            self.p = 90

    @ torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        result = []

      
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            _, feature = self.model.get_feature(images)
            
            output = ash_s_2d(feature, self.p)

            output = self.linear(output)
            output = self.T * torch.logsumexp(output / self.T, dim=1).data.cpu().numpy()

            result.append(output)

        return np.concatenate(result)



def ash_b(x, percentile):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x


def ash_p(x, percentile):
    assert x.dim() == 4
    assert 0 <= percentile <= 100

    b, c, h, w = x.shape

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    return x


def ash_s(x, percentile):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    s2 = x.sum(dim=[1, 2, 3])

    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x

def ash_s_2d(x, percentile=90):
    assert x.dim() == 2
    assert 0 <= percentile <= 100
    b, c = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=1)
    n = x.shape[1]
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=1)

    # apply sharpening
    scale = s1 / s2
    x = x * torch.exp(scale[:, None])

    return x

def ish(data, percentile):
    x = data.clone()
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    s2 = x.sum(dim=[1, 2, 3])

    scale = s1 / s2
    x = data * torch.exp(scale[:, None, None, None])

    return x