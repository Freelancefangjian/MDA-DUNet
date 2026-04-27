import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from DataSet import DataSet
from config import FLAGES
import torch.utils.data as Data
def l2_penaalty(w):
    return (w**2).sum()/2
import numpy as np
import time
import Model
import math
def PSNR(img1, img2):
    mse_sum  = (img1  - img2 ).pow(2)
    mse_loss = mse_sum.mean(2).mean(2)
    mse = mse_sum.mean()                     #.pow(2).mean()
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    # print(mse)
    return mse_loss, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
now = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
if __name__ == '__main__':
    #freeze_support()
    dataset = DataSet(FLAGES.pan_size, FLAGES.ms_size, FLAGES.img_path, FLAGES.data_path, FLAGES.batch_size,
                      FLAGES.stride)
    #HR = np.transpose(dataset.gt, [3, 1, 2])
    MSI = torch.from_numpy(np.transpose(dataset.pan, [0,3, 1, 2]))
    HSI = torch.from_numpy(np.transpose(dataset.ms, [0,3, 1, 2]))
    GT = torch.from_numpy(np.transpose(dataset.gt, [0,3, 1, 2]))

    torch_dataset = Data.TensorDataset(MSI, HSI, GT)
    loader = Data.DataLoader(dataset = torch_dataset, batch_size=64, shuffle=True, num_workers=2)

    device = torch.device("cuda:0")
    net = Model.Net()
    #net.load_state_dict(torch.load("./model/state_dicr_1000.pkl"))
    print('# generator parameters:', sum(param.numel() for param in net.parameters()))
    net.to(device)
    import torch.optim as optim

    criterion = nn.L1Loss().to(device)
    WEIGHT_DECAY = 1e-8
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=WEIGHT_DECAY)
    min_loss = 1.0
    for epoch in range(1001):  # loop over the dataset multiple times

        running_loss = 0.0
        mpsnr = 0.0

        for i, data in enumerate(loader, 0):
            # get the inputs
            MSI1, HSI1, GT1 = data
            MSI1 = MSI1.type(torch.FloatTensor)
            HSI1 = HSI1.type(torch.FloatTensor)
            GT1 = GT1.type(torch.FloatTensor)

            MSI1 = MSI1.cuda(device)
            HSI1 = HSI1.cuda(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(MSI1, HSI1)
            GT1 = GT1.cuda(device)
            loss = criterion(outputs, GT1)
            mse, psnr = PSNR(outputs, GT1)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            mpsnr += psnr
        if epoch % 10 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.7f PSNR:%.3f' %
                    (epoch + 1, i + 1, running_loss/(i + 1), mpsnr/(i + 1)))
            running_loss = 0.0
            if min_loss >= (running_loss/(i+1)):
                min_loss = (running_loss / (i + 1))
                torch.save(net.state_dict(), './model/better_state_dicr.pkl')
        if epoch % 100 == 0:  # print every 2000 mini-batches
            torch.save(net.state_dict(), './model/state_dicr_{}.pkl'.format(epoch))

    print('Finished Training')