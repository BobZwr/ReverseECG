import argparse
import numpy as np

from tqdm import tqdm
from net1d import *
from dataset import MyDataset
from torch.utils.data import DataLoader
import torch.optim as optim

def train(model, x, y, lr, batch_size, model_path):
    criterion = nn.BCEWithLogitsLoss()
    dataloader = DataLoader(MyDataset(x, y), batch_size=batch_size, shuffle=True)

    n_epoch = 100

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # You can add your scheduler here to adjust the learning rate.
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')


    prog = tqdm(range(n_epoch), desc="epoch", leave=False)
    model.train()
    for _ in prog:
        for batch_idx, batch in dataloader:
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = criterion(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # You can also add validation process if you want.
        # We only show the simplest case here.

    torch.save(model.cpu(), model_path + 'PretrainedModel.pkl')
    return

def generate_reverse(oridata):
    mu = np.mean(oridata, -1, keepdims=True)
    spatial_reverse = -1 * (oridata - mu) + mu
    temporal_reverse = oridata[:, ::-1]
    ts_reverse = spatial_reverse[::-1]
    x = np.concatenate((oridata, spatial_reverse, temporal_reverse, ts_reverse))
    n = oridata.shape[0]
    y = np.array([0, 0] * n + [0, 1] * n + [1, 0] * n + [1, 1] * n)

    return x, y
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_path',type=str,help='The path of your data')
    parser.add_argument('--model_saving_path',type=str,help='The path where you will save the model')
    parser.parse_args()

    original_data = np.load(parser.data_path)
    # The shape of original data should be [number of segments, length]
    x, y = generate_reverse(original_data)

    model_path = parser.model_saving_path
    # The model can be adjusted by yourself.
    # This is only an example.
    model = Net1D(in_channels=1,
                  base_filters=16,
                  ratio=1,
                  filter_list=[16, 32, 32, 64, 64],
                  m_blocks_list=[2, 2, 2, 2, 2],
                  kernel_size=16,
                  stride=2,
                  groups_width=16,
                  verbose=False,
                  use_bn=True,
                  n_classes=2).to(device)
    train(model, x, y, 1e-2, 128, model_path)