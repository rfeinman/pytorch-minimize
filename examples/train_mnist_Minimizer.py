import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

from torchmin import Minimizer


def MLPClassifier(input_size, hidden_sizes, num_classes):
    layers = []
    for i, hidden_size in enumerate(hidden_sizes):
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        input_size = hidden_size
    layers.append(nn.Linear(input_size, num_classes))
    layers.append(nn.LogSoftmax(-1))

    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model):
    train_output = model(X_train)
    test_output = model(X_test)
    train_loss = F.nll_loss(train_output, y_train)
    test_loss = F.nll_loss(test_output, y_test)
    print('Loss (cross-entropy):\n  train: {:.4f}  -  test: {:.4f}'.format(train_loss, test_loss))
    train_accuracy = (train_output.argmax(-1) == y_train).float().mean()
    test_accuracy = (test_output.argmax(-1) == y_test).float().mean()
    print('Accuracy:\n  train: {:.4f}  -  test: {:.4f}'.format(train_accuracy, test_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnist_root', type=str, required=True,
                        help='root path for the MNIST dataset')
    parser.add_argument('--method', type=str, default='newton-cg',
                        help='optimization method to use')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device to use for training')
    parser.add_argument('--quiet', action='store_true',
                        help='whether to train in quiet mode (no loss printing)')
    parser.add_argument('--plot_weight', action='store_true',
                        help='whether to plot the learned weights')
    args = parser.parse_args()

    device = torch.device(args.device)


    # --------------------------------------------
    #            Load MNIST dataset
    # --------------------------------------------

    train_data = datasets.MNIST(args.mnist_root, train=True)
    X_train = (train_data.data.float().view(-1, 784) / 255.).to(device)
    y_train = train_data.targets.to(device)

    test_data = datasets.MNIST(args.mnist_root, train=False)
    X_test = (test_data.data.float().view(-1, 784) / 255.).to(device)
    y_test = test_data.targets.to(device)


    # --------------------------------------------
    #           Initialize model
    # --------------------------------------------
    mlp = MLPClassifier(784, hidden_sizes=[50], num_classes=10)
    mlp = mlp.to(device)

    print('-------- Initial evaluation ---------')
    evaluate(mlp)


    # --------------------------------------------
    #         Fit model with Minimizer
    # --------------------------------------------
    optimizer = Minimizer(mlp.parameters(),
                          method=args.method,
                          tol=1e-6,
                          max_iter=200,
                          disp=0 if args.quiet else 2)

    def closure():
        optimizer.zero_grad()
        output = mlp(X_train)
        loss = F.nll_loss(output, y_train)
        # loss.backward()  <-- do not call backward!
        return loss

    loss = optimizer.step(closure)

    # --------------------------------------------
    #          Evaluate fitted model
    # --------------------------------------------
    print('-------- Final evaluation ---------')
    evaluate(mlp)

    if args.plot_weight:
        weight = mlp[0].weight.data.cpu().view(-1, 28, 28)
        vmin, vmax = weight.min(), weight.max()
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        axes = axes.ravel()
        for i in range(len(axes)):
            axes[i].matshow(weight[i], cmap='gray', vmin=0.5 * vmin, vmax=0.5 * vmax)
            axes[i].set_xticks(())
            axes[i].set_yticks(())
        plt.show()