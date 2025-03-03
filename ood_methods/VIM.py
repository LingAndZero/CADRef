import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils.linear_mapping import get_linear_layer_mapping
from scipy.linalg import pinv,norm
from scipy.special import logsumexp, softmax
from sklearn.covariance import EmpiricalCovariance


class VIM:

    def __init__(self, model, args,device, mode="VIM"):
        self.model = model
        self.device = device
        self.linear = get_linear_layer_mapping(args.model, args.ind_dataset, model)
        if mode not in ["VIM", "Residual"]:
            raise ValueError(f"Invalid mode: {mode}. Please choose from 'VIM' or 'Residual'.")
        else:   
            self.mode = mode # VIM or Residual
        
        self.NS = None
        self.alpha = None
        self.u = None

    @torch.no_grad()
    def get_ns(self,features):
        w = self.linear.weight.data
        b = self.linear.bias.data if self.linear.bias is not None else torch.zeros(w.size(0))
        w = w.cpu().numpy()
        b = b.cpu().numpy()
        u = -np.matmul(pinv(w), b)
        train_all_features = []
        for i in range(len(features)):
            train_all_features.extend(features[i].cpu().numpy())
        logit_id_train = train_all_features @ w.T + b
        if features[0].shape[-1] >= 2048:
            DIM = 1000
        elif features[0].shape[-1] >= 768:
            DIM = 512
        else:
            DIM = features[0].shape[-1] // 2
        print(f'{DIM=}')
        print('computing principal space...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(train_all_features - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)

        NS =  np.ascontiguousarray(
            (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

        print('computing alpha...')
        vlogit_id_train = norm(np.matmul(train_all_features - u, NS), axis=-1)
        alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
        print(f'{alpha=:.4f}')

        return NS, alpha,u

    def set_state(self, NS, alpha, u):
        self.NS = NS
        self.alpha = alpha
        self.u = u

    @ torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        result = []

        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)

            logit, feature = self.model.get_feature(images)
            feature = feature.cpu().numpy()
            logit = logit.cpu().numpy()

            vlogit =  norm(np.matmul(feature - self.u, self.NS), axis=-1) * self.alpha
            score = -vlogit
            if self.mode == "VIM": # Residual do not use energy
                energy = logsumexp(logit, axis=-1)
                score = score + energy
            result.append(score)

        return np.concatenate(result)
