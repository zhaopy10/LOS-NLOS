import torch.optim as optim
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model import Model


def main():
    print('Train....')
    model = Model(K=288, Tx=8, Rx=2)
    for name, param in model.named_parameters():
        print('Name:', name, 'Size:', param.size())
       
    batch_size = 8
    in_tensor = torch.randn(batch_size, 4, 288, 8, requires_grad=True)
    gt_labels = torch.empty(batch_size).random_(2)
    out_tensor = model(in_tensor)
    print('out size:', out_tensor.size())
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    loss = criterion(out_tensor, gt_labels)
    loss.backward()
    optimizer.step()
    print(loss.item())
    
if __name__ == '__main__':
    main()