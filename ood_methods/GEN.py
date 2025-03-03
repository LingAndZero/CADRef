import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


class GEN:

    def __init__(self, model, args,device):
        self.model = model
        self.device = device

        '''
        Special Parameters:
            T--Temperature
        '''
        self.gamma = 0.1
       
        if args.ind_dataset in ['cifar10', 'cifar100']:
            self.M = 10
        else:
            self.M = 100

    @ torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        result = []

        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            output = self.model(images)
            smax = (F.softmax(output, dim=1)).data.cpu().numpy()
            probs_sorted = np.sort(smax, axis=1)[:,-self.M:]
            scores = np.sum(probs_sorted ** self.gamma * (1 - probs_sorted) ** self.gamma, axis=1)

            result.append(-scores)

        return np.concatenate(result)
