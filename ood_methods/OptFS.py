import torch
import numpy as np
from tqdm import tqdm
from utils.linear_mapping import get_linear_layer_mapping

class OptFS:

    def __init__(self, model, args,device):
        self.model = model
        self.device = device
        self.linear = get_linear_layer_mapping(args.model, args.ind_dataset, model)
        self.theta = None
        self.left_boundary = None
        self.width = None
    

    def set_state(self, theta, left_boundary, width):
        self.theta = theta
        self.left_boundary = left_boundary
        self.width = width
    
    @torch.no_grad()
    def get_optimal_shaping(self, features, logits):
        features = torch.cat(features)
        logits = torch.cat(logits)

        preds = torch.softmax(logits, dim=1)
        features = features.cpu().numpy()
        preds = preds.argmax(dim=1).cpu().numpy()


        w = self.linear.weight.data
        b = self.linear.bias.data if self.linear.bias is not None else torch.zeros(w.size(0))
        w = w.cpu().numpy()
        b = b.cpu().numpy()

        left_b = np.quantile(features, 1e-3)
        right_b = np.quantile(features, 1-1e-3)
        
        width = (right_b - left_b) / 100.0
        left_boundary = np.arange(left_b, right_b, width)
        
        lc = w[preds] * features
        lc_fv_list = []
        for b in tqdm(left_boundary):
            mask = (features >= b) & (features < b + width)
            feat_masked = mask * lc
            res = np.mean(np.sum(feat_masked, axis=1))
            lc_fv_list.append(res)
        lc_fv_list = np.array(lc_fv_list)
        theta = lc_fv_list / np.linalg.norm(lc_fv_list, 2) * 1000

        theta = torch.from_numpy(theta[np.newaxis, :])
        return theta, left_boundary, width
    
    @torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        result = []

        with torch.no_grad():
            for (images, _) in tqdm(data_loader):
                images = images.to(self.device)
                _, feature = self.model.get_feature(images)
                feature = feature.view(feature.size(0), -1)

                feat_p = torch.zeros_like(feature).to(self.device)
                for i, x in enumerate(self.left_boundary):
                    mask = (feature >= x) & (feature < x + self.width)
                    feat_p += mask * feature * self.theta[0][i]

                output = self.linear(feat_p)
                output = torch.logsumexp(output , dim=1).data.cpu().numpy()

                result.append(output)

        return np.concatenate(result)
