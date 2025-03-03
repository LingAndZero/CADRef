import torch
import numpy as np
from tqdm import tqdm
from utils.linear_mapping import get_linear_layer_mapping

class DICE:

    def __init__(self, model, args,device):
        self.model = model
        self.device = device
        self.linear = get_linear_layer_mapping(args.model, args.ind_dataset, model)
        self.mask = None
        '''
        Special Parameters:
            T--Temperature
            p--Sparsity Parameter
        '''
        self.T = 1
        
        if args.ind_dataset in ['cifar10', 'cifar100']:
            self.p = 90
        else:
            self.p = 70

    def set_state(self, mask):
        self.mask = mask

    def get_mask(self, features):
        train_all_features = []
        for i in range(len(features)):
            train_all_features.extend(features[i].cpu().numpy())
        mean_feature = np.mean(train_all_features, axis=0)
        contrib = mean_feature[None, :] * self.linear.weight.data.squeeze().cpu().numpy()
        thresh = np.percentile(contrib, self.p)
        mask = torch.Tensor((contrib > thresh)).to(self.device)
        return mask

    @ torch.no_grad()
    def eval(self, data_loader):
        self.linear.weight.data *= self.mask
        self.model.eval()
        
        result = []

        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            output = self.model(images)
        
            output = self.T * torch.logsumexp(output / self.T, dim=1).data.cpu().numpy()

            result.append(output)

        return np.concatenate(result)
