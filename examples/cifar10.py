"""CIFAR10 example for cnn_finetune.
Based on:
- https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
- https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import argparse
import os
import csv
import shutil
from contextlib import redirect_stdout

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from cnn_finetune import make_model

parser = argparse.ArgumentParser(description='cnn_finetune cifar 10 example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-name', type=str, default='resnet50', metavar='M',
                    help='model name (default: resnet50)')
parser.add_argument('--dropout-p', type=float, default=0.2, metavar='D',
                    help='Dropout probability (default: 0.2)')
parser.add_argument('--log_dir', type=str, default=None,
                    help='Directory to save log')
parser.add_argument('--desc', type=str, default='',
                    help='description of test')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def train(model, epoch, optimizer, train_loader, criterion=nn.CrossEntropyLoss()):
    total_loss = 0
    total_size = 0
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()
        loss = criterion(output, target)
        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss / total_size))

    return {'train_loss': total_loss/total_size, 'train_acc': correct / len(train_loader.dataset)}


def test(model, test_loader, criterion=nn.CrossEntropyLoss()):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * acc))

    return {"test_loss": test_loss, "acc": acc}


def main():
    '''Main function to run code in this script'''

    model_name = args.model_name

    if model_name == 'alexnet':
        raise ValueError('The input size of the CIFAR-10 data set (32x32) is too small for AlexNet')

    classes = (
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )

    model = make_model(
        model_name,
        pretrained=True,
        num_classes=len(classes),
        dropout_p=args.dropout_p,
        input_size=(32, 32) if model_name.startswith(('vgg', 'squeezenet')) else None,
    )
    model = model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=model.original_model_info.mean,
            std=model.original_model_info.std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, args.test_batch_size, shuffle=False, num_workers=2
    )

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Use exponential decay for fine-tuning optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.975)


    # create log dir and files
    if (args.log_dir is None):
        model_dir = os.path.join(os.getcwd(), "logs", "cifar10", args.model_name, args.desc)
    else:
        model_dir = args.log_dir
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "train_log.csv"), "w") as train_log_file:
        train_log_csv = csv.writer(train_log_file)
        train_log_csv.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

    with open(os.path.join(model_dir, 'command_args.txt'), 'w') as command_args_file:
        for arg, value in sorted(vars(args).items()):
            command_args_file.write(arg + ": " + str(value) + "\n")

    with open(os.path.join(model_dir, 'model.txt'), 'w') as model_txt_file:
        with redirect_stdout(model_txt_file):
            print(model)

    # Train
    is_best = False
    best_acc1 = 0
    for epoch in range(1, args.epochs + 1):
        # Decay Learning Rate
        scheduler.step(epoch)
        train_log = train(model, epoch, optimizer, train_loader)
        test_log = test(model, test_loader)

        # append to log
        with open(os.path.join(model_dir, "train_log.csv"), "a") as train_log_file:
            train_log_csv = csv.writer(train_log_file)
            train_log_csv.writerow(((epoch,) + tuple(train_log.values()) + tuple(test_log.values()))) 


        acc1 = test_log["acc"]
        if acc1 > best_acc1:
            best_acc1 = acc1
            is_best = True
        else:
            is_best = False

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.model_name,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'lr_scheduler' : scheduler,
        }, is_best, model_dir)

        if(is_best):
            torch.save(model, os.path.join(model_dir, "model.pth"))
            torch.save(model.state_dict(), os.path.join(model_dir, "weights.pth"))

        # save weights
        os.makedirs(os.path.join(model_dir, 'weights_logs'), exist_ok=True)
        with open(os.path.join(model_dir, 'weights_logs', 'weights_log_' + str(epoch) + '.txt'), 'w') as weights_log_file:
            with redirect_stdout(weights_log_file):
                # Log model's state_dict
                print("Model's state_dict:")
                # TODO: Use checkpoint above
                for param_tensor in model.state_dict():
                    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
                    print(model.state_dict()[param_tensor])
                    print("")


def save_checkpoint(state, is_best, dir_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(dir_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(dir_path, filename), os.path.join(dir_path, 'checkpoint_best.pth.tar'))

    if (state['epoch']-1)%10 == 0:
        os.makedirs(os.path.join(dir_path, 'checkpoints'), exist_ok=True)
        shutil.copyfile(os.path.join(dir_path, filename), os.path.join(dir_path, 'checkpoints', 'checkpoint_' + str(state['epoch']-1) + '.pth.tar'))    


if __name__ == '__main__':
    main()
