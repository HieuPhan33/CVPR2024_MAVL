import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import numpy as np

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor, mask=None) -> torch.Tensor:
        '''
        x: B x n_disease
        target: B x n_ndisease
        '''
        if mask is not None:
            x[~mask] = float('-inf')
            sum_ = mask.sum(dim=-1, keepdim=True)
        logits = F.log_softmax(x, dim=-1)
        B = x.shape[0]
        if mask is not None:
            loss = torch.div(-target * logits, sum_)[mask]
        else:
            loss = (-target * logits).mean(dim=-1)
        loss = loss.sum() / B
        return loss
    


def pairwise_loss(outputs1, outputs2, label1, label2, sigmoid_param=1.):
    '''
    outputs: N_disease x B x concepts
    labels: N_disease x B
    '''
    #similarity = Variable(torch.mm(label1.data.float(), label2.data.float().t()) > 0).float()
    # outputs1 - N_disease x B x N_concepts
    similarity = Variable(torch.bmm(label1[:,:,None], label2[:,None,:])) # N_disease x B x B
    dot_product = sigmoid_param * torch.bmm(outputs1, outputs2.permute(0, 2, 1)) # N_disease x B x B
    exp_product = torch.exp(dot_product)

    exp_loss = (torch.log(1 + exp_product) - similarity * dot_product)
    mask_positive = similarity > 0
    mask_negative = similarity <= 0
    S1 = torch.sum(mask_positive.float())
    S0 = torch.sum(mask_negative.float())
    S = S0+S1

    exp_loss[similarity > 0] = exp_loss[similarity > 0] * (S / S1)
    exp_loss[similarity <= 0] = exp_loss[similarity <= 0] * (S / S0)

    loss = torch.mean(exp_loss)

    return loss









if __name__ == '__main__':
    N_disease, N_concepts, B, d = 10, 9, 4, 256
    x = torch.rand(N_disease, N_concepts, B, d)
    y = torch.randint(0, 2, (N_disease, B))
    loss = SupConLoss()
    loss(x, y)