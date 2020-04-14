from loss_fn import *
import torch.nn.functional as F
from torch.distributed.rpc import RRef
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer


class Trainer(object):

    def __init__(self, model, optimizer, train_loader, test_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train()
            #test_loss, test_acc = self.evaluate()

            print(
                'Epoch: {}/{},'.format(epoch, epochs),
                'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                #'test loss: {}, test acc: {}.'.format(test_loss, test_acc),
            )

    def train(self):

        train_loss = Average()
        train_acc = Accuracy()

        for data, target in self.train_loader:
            with dist_autograd.context() as context_id:
                data_ref = RRef(data)

                output_ref = self.model(data_ref)
                output = output_ref.to_here()
                loss = F.cross_entropy(output, target)

                dist_autograd.backward(context_id,[loss])
                self.optimizer.step(context_id)

                train_loss.update(loss.item(), data.size(0))
                train_acc.update(output, target)

        return train_loss, train_acc

    def evaluate(self):
        self.model.eval()

        test_loss = Average()
        test_acc = Accuracy()

        with torch.no_grad():
            for data, target in self.test_loader:
                with dist_autograd.context() as context_id:
                    data_ref = RRef(data)

                    output_ref = self.model(data_ref)
                    output = output_ref.to_here()
                    loss = F.cross_entropy(output, target)

                    test_loss.update(loss.item(), data.size(0))
                    test_acc.update(output, target)

        return test_loss, test_acc