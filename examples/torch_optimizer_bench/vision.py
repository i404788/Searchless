import torch
if torch.multiprocessing.get_start_method(allow_none=True) is None:
    torch.multiprocessing.set_start_method('spawn')
import torchvision
import torchvision.transforms as transforms
# import torch.nn as nn
import torch.nn.functional as F
from aim import Run
from pytorch_optimizer import create_optimizer
from searchless import LinearFloatProxy, LogFloatProxy, EnumeratorProxy, resolve_proxies

device = 'cuda'
torch.set_default_device(device)

samples_per_epoch = 10000
hp = {'epochs': 16}
hp['optimizer'] = EnumeratorProxy(['AdaBelief', 'AdaBound', 'AdamP', 'Adan', 'Lamb', 'MADGRAD', 'DAdaptAdan', 'Lion', 'NovoGrad', 'Gravity', 'Apollo', 'RAdam']) # TODO: add more
hp['lr'] = LogFloatProxy(1e-5, 5e-2, 6)
# hp['weight_decay'] = LogFloatProxy(1e-6, 1e-4, 3)
hp['batch_size'] = 128 #EnumeratorProxy([4, 16, 64, 256])
hp['clip'] = LinearFloatProxy(0.5, 1, 2)
hp['seed'] = EnumeratorProxy([42, 0xFFFF-42])
resolve_proxies(hp)

torch.manual_seed(hp['seed'])

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=hp['batch_size'], shuffle=True, num_workers=2, generator=torch.Generator(device=device))

testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=hp['batch_size'], shuffle=False, num_workers=2, generator=torch.Generator(device=device))


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

run = Run(experiment='vision-cifar10')
run['hparams'] = hp
net = Net()

opt = create_optimizer(net, str(hp['optimizer']), lr=float(hp['lr']))#, weight_decay=hp['weight_decay'])
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(hp['epochs']):  # loop over the dataset multiple times
    running_loss = 0.0
    i = 0
    for data in trainloader:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = map(lambda v: v.to(device), data)

        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), hp['clip'])
        opt.step()

        # print statistics
        run.track(loss.item(), name='loss', context={'subset': 'train'}, step=i)
        run.track(torch.cuda.memory_reserved(device), name='mem_res', context={'subset': 'train'}, step=i)
        run.track(torch.cuda.memory_allocated(device), name='mem_alloc', context={'subset': 'train'}, step=i)
        running_loss += loss.item()
        if i//hp['batch_size'] % samples_per_epoch == samples_per_epoch-1:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

        i += hp['batch_size']

# TODO: store top10 test/train accuracy, top1 test/train accuracy
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        inputs, labels = map(lambda v: v.to(device), data)
        # calculate outputs by running images through the network
        outputs = net(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
run.track(correct/total, name='accuracy', context={'subset': 'test'}, step=i)
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
print(hp)