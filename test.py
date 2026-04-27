import scipy.io as sio
import numpy as np
import Model
import os
import torch
if __name__ == '__main__':
    net = Model.Net().cuda()
    net.load_state_dict(torch.load("./model/state_dicr_800.pkl"))
    test_path = 'G:\\OneDrive - njust.edu.cn\\Hyperspectral_Image_Benchmarkx8\\harvard\\MW-DAN\\test'
    for i in range(20):
        ind = i + 1
        path = str(i + 1) + '.mat'
        print('processing for %d' % ind)
        source_hs_path = os.path.join(test_path, 'hs', path)
        data = sio.loadmat(source_hs_path)
        data = torch.FloatTensor(data['I']).permute(2,0,1).unsqueeze(0).cuda()
        source_ms_path = os.path.join(test_path, 'ms', path)
        data1 = sio.loadmat(source_ms_path)
        data1 = torch.FloatTensor(data1['I']).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            data_get = net(data1, data)
        data_get = data_get.cpu().detach().numpy()
        data_get = np.transpose(data_get, [0, 2, 3, 1])
        data_get = np.reshape(data_get, (1024, 1024, 31))
        data_get = np.array(data_get, dtype=np.float64)
        sio.savemat('./get/eval_%d.mat' % ind, {'b': data_get})
        torch.cuda.empty_cache()