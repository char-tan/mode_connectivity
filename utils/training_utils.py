import torch.nn.functional as F
import torch


def train(
    args, model, device, train_loader, optimizer, epoch, verbose: int = 2
):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            acc = 100.0 * correct / len(train_loader.dataset)
            if verbose >= 2:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    acc = 100.0 * correct / len(train_loader.dataset)
    if verbose >= 1:
        print("Train Epoch: {}, Train Accuracy: ({:.0f}%) ".format(epoch, acc))


def test(model, device, test_loader, verbose: int = 2):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100.0 * correct / len(test_loader.dataset)
    if verbose >= 1:
        print("Average loss: {:.4f}, Accuracy: ({:.0f}%)".format(test_loss, acc))
    return test_loss, acc
