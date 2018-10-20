using PyCall

@pyimport torchvision
@pyimport torchvision.transforms as transforms
@pyimport torchvision.datasets as datasets


function getDataLoaders(trainData, testData, batchsize)
    trainLoader = torch.utils[:data][:DataLoader](dataset=trainData,
                                                  batch_size=batchsize,
                                                  shuffle=true)
    testLoader = torch.utils[:data][:DataLoader](dataset=testData,
                                                 batch_size=batchsize,
                                                 shuffle=false)
    trainLoader, testLoader
end

function getmnistDataLoaders(batchsize)
    trainData = datasets.MNIST(root="../data",
                               train=true,
                               transform=transforms.ToTensor(),
                               download=true)
    testData = datasets.MNIST(root="../data",
                              train=false,
                              transform=transforms.ToTensor())

    getDataLoaders(trainData, testData, batchsize)
end

function getcifar10DataLoaders(batchsize)
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])

    trainData = datasets.CIFAR10(root="../data",
                                 train=true,
                                 transform=transform,
                                 download=true)
    testData = datasets.CIFAR10(root="../data",
                                train=false,
                                transform=transforms.ToTensor())
    getDataLoaders(trainData, testData, batchsize)
end

#trainLoader, testLoader = getcifar10DataLoaders(128)
#x = trainLoader[:dataset][1] |> first
#x[:shape]
