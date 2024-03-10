import torch.nn
import argparse
from Server import *

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
sys.path.append("../..")

cudaIdx = "cuda:0"  # GPU card index
device = torch.device(cudaIdx if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=500, help="total epochs")
parser.add_argument("--local_epochs", default=10,help="The config file")
parser.add_argument("--saveLossInterval", default=1,help="intervals to save loss")
parser.add_argument("--saveModelInterval", default=10,help="intervals to save model")
parser.add_argument("--batchSize", default=16,help="batchsize for training and evaluation")
parser.add_argument("--num_users", default=90,help="total users")
parser.add_argument("--num_activate_users", default=5,help="activate users")
parser.add_argument("--lr", default=4e-3,help="learning rate")
parser.add_argument("--num_workers", default=0,help="The num of workers")
parser.add_argument("--evaluation", default=False,help="evaluation only if True")
parser.add_argument("--Result", default='results',help="dir name of saving file")
parser.add_argument("--key_info", default="Try_001",help="name of saving file")
args = parser.parse_args()

def train_main(train_dataset_path):
    work_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件目录
    Result = os.path.join(work_dir, 'results')
    list = ['Global Epoch', 'loss', 'pathloss_score']
    data = pd.DataFrame([list])  # 创建DataFrame数据结构框架
    data.to_csv(os.path.join(Result, args.key_info+'.csv'), mode='w', header=None, index=False)
    seed_everything(42)
    train_datasets = []
    valid_datasets = []
    train_dataloaders = []
    if not os.path.exists(f'models/'):
        os.makedirs(f'models/')
    if not os.path.exists(f'results/'):
        os.makedirs(f'results/')
    # for start_user in range(11,85,10):
    #     end_user = start_user + 9

    for i in range(1, 90 + 1):
        all_dataset = DoraSet(train_dataset_path, set='train', clientId=i)
        train_size = int(0.99 * len(all_dataset))
        valid_size = len(all_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(all_dataset, [train_size, valid_size])
        train_datasets.append(train_dataset)
        valid_datasets.append(valid_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, args.batchSize, shuffle=True, num_workers=args.num_workers)
        train_dataloaders.append(train_loader)

    print("length of train_dataloaders:",len(train_dataloaders))
    valid_data_comb = DoraSetComb(valid_datasets)
    print("length of valid_data_comb:",len(valid_data_comb))
    valid_loader = torch.utils.data.DataLoader(valid_data_comb, 1, shuffle=False, num_workers=args.num_workers)
    print("length of valid_loader:",len(valid_loader))

    model = DoraNet()
    # model.load_state_dict(torch.load("models/model500.pth"))
    server = FedAvgServer(args, model, train_dataloaders, valid_loader)
    server.train()


if __name__ == '__main__':
    train_main("data/train/")
