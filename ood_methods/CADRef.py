import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils.linear_mapping import get_linear_layer_mapping

class CADRef:

    def __init__(self, model, args,device):
        self.model = model
        self.device = device
        self.logit_method = args.logit_method
        self.linear = get_linear_layer_mapping(args.model, args.ind_dataset, model)
        self.train_mean = None
        self.global_mean_logit_score = None
    def set_state(self, train_mean, global_mean_logit_score):
        self.train_mean = train_mean
        self.global_mean_logit_score = global_mean_logit_score

    @torch.no_grad()
    def get_mean_feature(self, features):
        train_mean = []
        for i in range(len(features)):
            train_mean.append(features[i].mean(dim=0))
        train_mean = torch.stack(train_mean)
        return train_mean
    
    @torch.no_grad()
    def get_global_mean_logit_score(self,logits):
        train_logit_score = []
        for i in range(len(logits)):
            train_logit_score.append(self.get_logits_score(logits[i]))
        mean_logit_score = torch.mean(torch.cat(train_logit_score),dim=0)
        return mean_logit_score
    
    @torch.no_grad()
    def get_state(self,features,logits):
        train_mean = self.get_mean_feature(features)
        global_mean_logit_score = self.get_global_mean_logit_score(logits)
        return train_mean, global_mean_logit_score
        
    def get_logits_score(self,logits):
        if self.logit_method == "MaxLogit":
            return maxLogits(logits)
        elif self.logit_method == "GEN":
            return gen(logits)
        elif self.logit_method == "Energy":
            return energy(logits)
        elif self.logit_method == "MSP":
            return msp(logits)

    @torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        result = []
        w = self.linear.weight.data
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            logit, feature = self.model.get_feature(images)
            class_ids = torch.argmax(torch.softmax(logit, dim=1), dim=1).cpu().numpy()
            tm = self.train_mean[class_ids].to(self.device)
            dist = feature - tm
            sg = w[class_ids].sign()
            ep_dist = dist * sg
            ep_dist[ep_dist<0] = 0
            ep_error = ep_dist.norm(dim=1,p=1)/feature.norm(dim=1,p=1)

            en_dist = dist*(-sg)
            en_dist[en_dist<0] = 0
            en_error = en_dist.norm(dim=1,p=1)/feature.norm(dim=1,p=1)

            logit_score = self.get_logits_score(logit)
            score = ep_error/logit_score + en_error/self.global_mean_logit_score

            result.append(-score.cpu().numpy())
        return np.concatenate(result)


def maxLogits(output):
    scores = output.max(dim=1).values
    return scores

def gen(output,gamma=0.1):
    M = output.shape[-1]//10
    M = 10 if M <10 else M
    smax = (F.softmax(output, dim=1))
    probs_sorted = torch.sort(smax, dim=1).values[:,-M:]
    scores = torch.sum(probs_sorted ** gamma * (1 - probs_sorted) ** gamma, axis=1)
    return 1/scores

def energy(output):
    return torch.logsumexp(output, dim=1)

def msp(output):
    return torch.max(F.softmax(output, dim=1), dim=1).values

