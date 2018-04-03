import torch
from torch import nn
import torch.nn.functional as F 


class CnnPolicy(nn.Module):
    def __init__(self, nb_actions):
        super(CnnPolicy, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=64*6*6, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=nb_actions)
                
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(-1, 64*6*6)
        x = self.relu(self.fc1(x))
        scores = self.fc2(x)
        return F.softmax(scores, dim=1)
    
    def gather_flat_params(self):
        views = []
        for p in self.parameters():
            view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def replace_params(self, new_params):
        """Replaces network parameters with the given ones.
           https://discuss.pytorch.org/t/vector-view-of-model-parameters/3783/2
        """
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = new_params[offset: offset+numel].view_as(p.data)
            offset += numel

    def gather_flat_grad(self):
        """"Gathers gradients from layers into a single flattened vector. 
            https://discuss.pytorch.org/t/vector-view-of-model-parameters/3783/2
        """
        views = []
        for p in self.parameters():
            if p.grad is None:
                view = p.new(p.data.numel()).zero_()
            elif p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)
        
