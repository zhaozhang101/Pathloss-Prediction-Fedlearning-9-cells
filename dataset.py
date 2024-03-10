import os
import numpy as np
from torch.utils.data import Dataset

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"


class DoraSet(Dataset):
    def __init__(self, dataset_path, set='train', clientId=1):
        self.set = set
        folder = dataset_path
        if set == 'test':
            self.data = np.reshape(np.load(folder + 'test.npy'), (-1, 6))
        else:
            self.data = np.reshape(np.load(folder + f'user_{clientId:02d}.npy'), (-1, 6))
        self.data[:, :2] = (self.data[:, :2] - np.array([162.69, 167.50])) / np.array([92.13, 91.21])
        new_idx = np.zeros_like(self.data[:,2:])
        new_idx[self.data[:,2:]==0] = 1

        self.data[:, 2:] = (self.data[:, 2:] - np.array([-83.63, -81.95, -84.59, -82.58])) / np.array([14.35, 18.06, 20.14, 21.26])
        self.data[:, 2:][new_idx==1] = 0


    def __getitem__(self, idx):
        # 162.69553466165152 92.12944541001944
        # 167.50507994638315 91.20744704249563
        # -83.63877986074748 14.356142334145595
        # -81.95750666978137 18.060331759145292
        # -84.59552483282998 20.14371746595895
        # -82.58517819098684 21.263828579168226

        return self.data[idx, :2], self.data[idx, 2:]

    def __len__(self):
        return len(self.data)


class DoraSetComb(Dataset):
    def __init__(self, datasets):
        self.dataLen = []
        self.datasets = datasets
        for i in datasets:
            self.dataLen.append(len(i))

    def __getitem__(self, idx):
        for i in range(len(self.dataLen)):
            if idx < np.sum(self.dataLen[:i + 1]):
                if i == 0:
                    idx2 = idx
                else:
                    idx2 = idx - np.sum(self.dataLen[:i])
                break
        return self.datasets[i][idx2]

    def __len__(self):
        return np.sum(self.dataLen)


if __name__ == '__main__':
    dataset = DoraSet("data/train/", set="train", clientId=1)
    pos = np.zeros(shape=(len(dataset), 2))
    pathloss = np.zeros(shape=(len(dataset), 4))
    for index in range(len(dataset)):
        pos[index], pathloss[index] = dataset[index]

    print(f'pos:', pos)
    print(f'pathloss:', pathloss)
