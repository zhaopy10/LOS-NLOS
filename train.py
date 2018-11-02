import torch.optim as optim
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from dataloader import Dataloader
from model import Model
import sys

def eval(model, loader):
    loader.reset()
    done = False
    LOS_to_LOS = 0
    LOS_to_NLOS = 0
    NLOS_to_LOS = 0
    NLOS_to_NLOS = 0
    while not done:
        channels, labels, SNR, done = loader.next_batch()
        out_tensor = model(channels)
        out_tensor_np = out_tensor.cpu().detach().numpy()
        gt_labels = labels.cpu().detach().numpy()
        for i in range(len(gt_labels)):
            if out_tensor_np[i] > 0.5:
                if gt_labels[i] == 1:
                    LOS_to_LOS += 1
                else:
                    NLOS_to_LOS += 1
            else:
                if gt_labels[i] == 1:
                    LOS_to_NLOS += 1
                else:
                    NLOS_to_NLOS += 1
        
    print("Accuracy: %.3f, Percision: %.3f, Recall: %.3f"%(float(LOS_to_LOS + NLOS_to_NLOS) / (LOS_to_LOS + LOS_to_NLOS + NLOS_to_LOS + NLOS_to_NLOS) ,float(LOS_to_LOS) / (LOS_to_LOS + NLOS_to_LOS),float(LOS_to_LOS) / (LOS_to_LOS + LOS_to_NLOS)))
    #raw_input()

def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    print(device)
    model = Model(K=288, Tx=8, Rx=2)
    model.to(device)
    for name, param in model.named_parameters():
        print('Name:', name, 'Size:', param.size())
        
    epoch = 100
    batch_size = 1024
    loader = Dataloader(path='/home/zpy/work/data/data_separate/training', batch_size=batch_size, device=device)
    eval_loader = Dataloader(path='/home/zpy/work/data/data_separate/test', batch_size=batch_size, device=device)
    
    
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
    for e in range(epoch):
        print('Train %d epoch'%(e))
        loader.reset()
        eval_loader.reset()
        done = False
        running_loss = 0
        batch_num = 0
        while not done:
            batch_num += 1
            channels, labels, SNR, done = loader.next_batch()
            out_tensor = model(channels)
            loss = criterion(out_tensor, labels)
            #print(loss.item())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #print('\rloss: %.3f', running_loss / batch_num)
        
        print('[%d] loss: %.3f' %
                  (e + 1, running_loss / batch_num))
        eval(model, eval_loader)
    
if __name__ == '__main__':
    main()