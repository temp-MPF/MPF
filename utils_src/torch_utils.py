import torch
import torch.nn.functional as F

def sample_gumbel(shape,device, eps=1e-20):
    U = torch.Tensor(shape).uniform_(0,1).to(device) #.cuda()
    return -(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(),device = logits.device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1, hard=False):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y = (y_hard - y).detach() + y
    return y
