using PyCall

@pyimport torchvision
@pyimport torchvision.transforms as transforms
@pyimport torchvision.datasets as datasets


function getDataLoaders(batchsize)
    trainData = datasets.MNIST(root="../data",
                               train=true,
                               transform=transforms.ToTensor(),
                               download=true)

    testData = datasets.MNIST(root="../data",
                              train=false,
                              transform=transforms.ToTensor())

    trainLoader = torch.utils[:data][:DataLoader](dataset=trainData,
                                                  batch_size=batchsize,
                                                  shuffle=true)

    testLoader = torch.utils[:data][:DataLoader](dataset=testData,
                                                 batch_size=batchsize,
                                                 shuffle=false)
    trainLoader, testLoader
end
