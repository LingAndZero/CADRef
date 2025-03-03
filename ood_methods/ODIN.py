import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm


class ODIN:

    def __init__(self, model, args,device):
        self.model = model
        self.device = device

        '''
        Special Parameters:
            T--Temperature
            epsilon--Perturbation Magnitude
        '''
        self.T = 1000
        if args.ind_dataset == 'ImageNet':
            self.epsilon = 0.005
        elif args.ind_dataset in ['cifar100']:
            self.epsilon = 0.002
        else:
            self.epsilon = 0.0014

    def inputPreprocessing(self, images):
        outputs = self.model(images)
        criterion = nn.CrossEntropyLoss(reduction="sum")

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

        # Using temperature scaling
        outputs = outputs / self.T

        labels = Variable(torch.LongTensor(maxIndexTemp).to(self.device))
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(images.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        gradient[:, 0] = (gradient[:, 0]) / (63.0 / 255.0)
        gradient[:, 1] = (gradient[:, 1]) / (62.1 / 255.0)
        gradient[:, 2] = (gradient[:, 2]) / (66.7 / 255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(images.data, gradient, alpha=-self.epsilon)
        with torch.no_grad():
            outputs = self.model(Variable(tempInputs))
        outputs = outputs / self.T

        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        return nnOutputs


    def eval(self, data_loader):
        self.model.eval()
        result = []

        for (images, _) in tqdm(data_loader):
            images = Variable(images.to(self.device), requires_grad=True)
            score = self.inputPreprocessing(images)
            score = np.max(score, 1)

            result.append(score)

        return np.concatenate(result)
