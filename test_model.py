
import torch
import model

from dataset import DoraSet, DoraSetComb
from model import DoraNet
cudaIdx = "cuda:0"
device = torch.device(cudaIdx if torch.cuda.is_available() else "cpu")
test_instances = 10
from util import *


path = "data/train/"
model = model.DoraNet().to(device)
model.load_state_dict(torch.load("models/model400.pth"))
model.eval()
train_dataloaders = []
train_datasets = []
valid_datasets = []
for i in range(1, 90 + 1):
    all_dataset = DoraSet("data/train/", set='train', clientId=i)
    train_size = int(0.99 * len(all_dataset))
    valid_size = len(all_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(all_dataset, [train_size, valid_size])
    train_datasets.append(train_dataset)
    valid_datasets.append(valid_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, 256, shuffle=True, num_workers=0)
    train_dataloaders.append(train_loader)

valid_data_comb = DoraSetComb(valid_datasets)
valid_loader = torch.utils.data.DataLoader(valid_data_comb, 1, shuffle=False, num_workers=0)
scores = 0
for i, (pos, pathloss) in enumerate(valid_loader):
    pos = pos.float().to(device)
    pathloss = pathloss.float().numpy()
    p_pathloss = model(pos).cpu().detach().numpy()
    # loss = torch.mean(torch.abs(p_pathloss[pathloss != 0] - pathloss[pathloss != 0]))  ## unit in dB
    print("{}:pos_{}  pathloss:{}  prepathloss:{}".format(i,pos, pathloss, p_pathloss))
    tmp1 = np.sum(
                np.abs(10 ** (0.1 * p_pathloss[pathloss != 0]) - 10 ** (0.1 * pathloss[pathloss != 0])) ** 2)
    tmp2 = np.sum(np.abs(10 ** (0.1 * pathloss[pathloss != 0])) ** 2)
    score = tmp1 / tmp2
    scores += -10 * np.log10(score)
print(scores)
    # if score > 1:
    #     score = torch.tensor([1])
    # losses.update(loss.item(), len(pos))
    # scores.update(score.item(), len(pos))

# for i in range(test_instances):
#     envPath = path
#     data = np.reshape(np.load(envPath + 'user_48.npy'), (-1, 6))
#     pos = torch.Tensor(data[[i], :2]).to(device)
#     pathloss = data[[i], 2:]
#     p_pathloss = model(pos).cpu().detach().numpy()
#     print("{}:  pathloss:{}  prepathloss:{}".format(i, pathloss, p_pathloss))
#     tmp1 = np.sum(
#         np.abs(10 ** (0.1 * p_pathloss[pathloss != 0]) - 10 ** (0.1 * pathloss[pathloss != 0])) ** 2)
#     tmp2 = np.sum(np.abs(10 ** (0.1 * pathloss[pathloss != 0])) ** 2)
#     score = tmp1 / tmp2
#     print(-10 * np.log10(score))
