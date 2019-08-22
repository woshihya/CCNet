import os
import argparse
import torch
import time
import torch.nn as nn
import torch.optim as optim
import global_settings
import shutil
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(epoch):
    net.train()
    correct = 0.0
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if epoch <= 1:
            warmup_scheduler.step()

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = outputs.max(1)
        correct_tmp = preds.eq(labels).sum()
        correct += correct_tmp
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    print('Train Epoch:{}, Average accuracy: {:.4f}， learning rate:{:.4f}'.format(
        epoch,
        correct.float() / len(cifar100_training_loader.dataset),
        optimizer.param_groups[0]['lr']
    ))


def eval_training(epoch):
    net.eval()
    total_size = 0
    correct_top1 = 0
    correct_top5 = 0
    for (images, labels) in cifar100_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        top5 = torch.sort(outputs, dim=1, descending=True)[1][:, :5]
        top1 = top5[:, 0]

        total_size += int(labels.size(0))
        correct_top1 += int((top1 == labels).sum().item())
        for i in range(top5.shape[0]):
            if labels[i] in top5[i]:
                correct_top5 += 1
    print("Test Epoch:{0}, top1 accuracy：{1}, top5 accuracy: {2}".format(epoch, correct_top1 / total_size, correct_top5 / total_size))
    return correct_top1 / total_size


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='CCNet149', help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    net = get_network(args)
    print('parameters:', sum(param.numel() for param in net.parameters()) / 1024 / 1024)
    if os.path.exists('checkpoint/net.pkl'):
        checkpoint = torch.load('checkpoint/net.pkl')
        net.load_state_dict(checkpoint['net'])
        print("checkpoint load success")
        print(checkpoint['epoch'])

    # data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        global_settings.CIFAR100_TRAIN_MEAN,
        global_settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    cifar100_test_loader = get_test_dataloader(
        global_settings.CIFAR100_TRAIN_MEAN,
        global_settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, dampening=0, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=global_settings.MILESTONES,
                                                     gamma=0.2)
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * 1)
    checkpoint_path = 'checkpoint/net.pkl'

    acc_all = []
    for epoch in range(1, global_settings.EPOCH):
        if epoch > 1:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)
        acc_all.append(acc)
        state = {'net': net.state_dict(), 'epoch': epoch}
        torch.save(state, 'checkpoint/net_{0}.pkl'.format(epoch))

    argmax = acc_all.index(max(acc_all))
    print(argmax, max(acc_all))
    shutil.copy('checkpoint/net_{0}.pkl'.format(argmax), 'checkpoint/net.pkl')
