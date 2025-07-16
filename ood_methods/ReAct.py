import torch
import numpy as np
from tqdm import tqdm
from utils.linear_mapping import get_linear_layer_mapping

class ReAct:

    def __init__(self, model, args,device):
        self.model = model
        self.device = device
        self.linear = get_linear_layer_mapping(args.model, args.ind_dataset, model)
        '''
        Special Parameters:
            T--Temperature
            p--Truncation Percentage
        '''
        self.T = 1
        self.p = 90

    def set_state(self, threshold):
        self.threshold = threshold

    @torch.no_grad()
    def get_threshold(self, features):
        train_all_features = []
        for i in range(len(features)):
            train_all_features.extend(features[i].cpu().numpy())
        threshold = np.percentile(train_all_features, self.p)
        print(threshold)
        return threshold

    @torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        result = []

        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            _, feature = self.model.get_feature(images)
            feature = feature.clip(max=self.threshold)
            output = self.linear(feature)

            output = self.T * torch.logsumexp(output / self.T, dim=1).data.cpu().numpy()

            result.append(output)

        return np.concatenate(result)
