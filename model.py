from torch import nn
import torch.nn.functional as F
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef

from torch.distributed.rpc import rpc_sync
from torch.distributed.rpc import rpc_async
#from torchsummary import summary


def _parameter_rrefs(module):
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs

def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)

class ConvNet(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5).to(self.device)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5).to(self.device)

    def forward(self, rref):
        t = rref.to_here().to(self.device)
        # conv 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # conv 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        return t.cpu()
    
class FCNet(nn.Module):

    def __init__(self,device):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120).to(self.device)
        self.fc2 = nn.Linear(in_features=120, out_features=60).to(self.device)
        self.out = nn.Linear(in_features=60, out_features=10).to(self.device)

    def forward(self, rref):
        # conv 1
        t = rref.to_here().to(self.device)

        # fc1
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = self.out(t)
        # don't need softmax here since we'll use cross-entropy as activation.

        return t.cpu()

class CNNModel(nn.Module):
    def __init__(self, connet_wk, fcnet_wk, device):
        super(CNNModel, self).__init__()

        # setup embedding table remotely
        self.device = device
        
        self.convnet_rref = rpc.remote(connet_wk, ConvNet,args=(device,))
        # setup LSTM locally
        print(self.convnet_rref.to_here())
        self.fcnet_rref = rpc.remote(fcnet_wk, FCNet,args=(device,))
        #print(self.fcnet_rref.to_here())
        print('CNN model constructed: ' + 'owner')


    def forward(self, inputreff):
        
        convnet_forward_rref = rpc.remote(self.convnet_rref.owner(), _call_method, args=(ConvNet.forward, self.convnet_rref, inputreff))
        
        fcnet_forward_rref = rpc.remote(self.fcnet_rref.owner(), _call_method, args=(FCNet.forward, self.fcnet_rref, convnet_forward_rref))
                                                                    
        return fcnet_forward_rref
    
    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(_remote_method(_parameter_rrefs, self.convnet_rref))
        remote_params.extend(_remote_method(_parameter_rrefs, self.fcnet_rref))
        return remote_params