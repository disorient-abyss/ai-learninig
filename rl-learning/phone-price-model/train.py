import os
import time

import torch
from PhonePriceModel import PhonePriceModel
from price_classify import create_dataset
from torch import nn, optim
from torch.utils.data import DataLoader


def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'正在使用的训练设备: {device}')

    train_dataset, _, input_dim, class_num = create_dataset()
    torch.manual_seed(0)

    model = PhonePriceModel(input_dim, class_num).to(device)
    criterion = nn.CrossEntropyLoss()

    # 1. 增大学习率至 1.5e-3，配合适当的 weight_decay
    num_epoch = 50
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)

    dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)  # 推荐 batch_size=16

    for epoch_idx in range(num_epoch):
        model.train()
        start = time.time()
        total_loss = 0.0
        total_num = 0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(y)
            total_num += batch_size
            total_loss += loss.item() * batch_size

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f'epoch: {epoch_idx + 1:2d} | loss: {total_loss / total_num:.4f} | lr: {current_lr:.6f} | time: {time.time() - start:.2f}s'
        )

    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), 'model/phone-price-model.bin')


if __name__ == '__main__':
    train()