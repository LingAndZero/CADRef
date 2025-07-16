from torch.autograd import Variable
import numpy as np
import torch
from tqdm import tqdm


class GradNorm:

    def __init__(self, model, args,device):
        self.model = model
        self.device = device
        self.num_classes = args.num_classes
        '''
        Special Parameters:
            T--Temperature
        '''
        self.T = 1

    def eval(self, data_loader):
        self.model.eval()
        result = []
        logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()

        for (images, _) in tqdm(data_loader):
            images = Variable(images.to(self.device), requires_grad=True)

            self.model.zero_grad()
            outputs = self.model(images)
            targets = torch.ones((images.shape[0], self.num_classes)).to(self.device)
            outputs = outputs / self.T

            loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))
            loss.backward()

            linear_grad = self.model.linear.weight.grad.data
            score = torch.sum(torch.abs(linear_grad)).item()

            result.append(score)

        return np.array(result)