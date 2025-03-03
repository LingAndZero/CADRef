import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


class CARef:

    def __init__(self, model, args,device):
        self.model = model
        self.device = device
        self.train_mean = None
    
    def set_state(self, train_mean):
        self.train_mean = train_mean

    @torch.no_grad()
    def get_mean_feature(self, features):
        train_mean = []
        for i in range(len(features)):
            train_mean.append(features[i].mean(dim=0))
        
        train_mean=torch.stack(train_mean)
        return train_mean
    
    @torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        result = []

        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            logit, feature = self.model.get_feature(images)
            class_ids = torch.argmax(torch.softmax(logit, dim=1), dim=1).cpu().numpy()
            tm = self.train_mean[class_ids].to(self.device)
            score = (tm-feature).norm(dim=1,p=1)/feature.norm(dim=1,p=1)
            result.append(-score.cpu().numpy())
        return np.concatenate(result)
