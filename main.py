import torch.distributed.rpc as rpc
import torch
import os
import torch.optim as optim
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torchvision import datasets, transforms
from train import *
from dataload import *
from model import *

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = CNNModel(args['host'], args['worker'],device)
    # setup distributed optimizer
    opt = DistributedOptimizer(
        optim.Adam,
        model.parameter_rrefs(),
        lr=args['lr'],
    )

    train_loader = MNISTDataLoader(args['root'], args['batch_size'], train=True)
    test_loader = MNISTDataLoader(args['root'], args['batch_size'], train=False)

    trainer = Trainer(model, opt, train_loader, test_loader, device)
    trainer.fit(args['epochs'])

    
def run_rpc(rank):
    argv = {'world_size': int(2),
            'rank': int(0),
            'host': "worker0",
            'worker': "worker1",
            'epochs': int(5),
            'lr': float(1e-3),
            'root': 'data',
            'batch_size': int(32)
           }
    
    if(rank == 0):
        print(argv)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29502'
        rpc.init_rpc(argv['host'], rank=rank, world_size=argv['world_size'])
        print('Start Run:', rank)
        run(argv)
        rpc.shutdown()
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29502'
        rpc.init_rpc(argv['worker'], rank=rank, world_size=argv['world_size'])
        print('Start Run:', rank)
        rpc.shutdown()

        
        
world_size = 2
processes = []
for rank in range(world_size):
    p = mp.Process(target=run_rpc, args=(rank,))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
