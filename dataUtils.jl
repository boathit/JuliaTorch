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


using Random:randperm!

struct DataLoader
    data::Tuple
    batchsize::Int
    shuffle::Bool
    indices::Vector{Int}
    n::Int
end

function DataLoader(data::NTuple{N, AbstractArray}; batchsize::Int=100, shuffle=false) where N
    lens = [last(size(x)) for x in data]
    n = first(lens)
    @assert all(len -> len == n, lens) "The last dimension of all the Arrays should be equal"
    # n = last(size(first(data)))
    indices = collect(1:n)
    shuffle && randperm!(indices)
    DataLoader(data, batchsize, shuffle, indices, n)
end

DataLoader(data...; batchsize::Int=100, shuffle=false) =
    DataLoader(data, batchsize=batchsize, shuffle=shuffle)

function Base.iterate(it::DataLoader, start=1)
    if start > it.n
        it.shuffle && randperm!(it.indices)
        return nothing
    end
    nextstart = min(start + it.batchsize, it.n + 1)
    i = it.indices[start:nextstart-1]
    element = Tuple(copy(selectdim(x, ndims(x), i)) for x in it.data)
    return element, nextstart
end

Base.length(it::DataLoader) = it.n
Base.eltype(it::DataLoader) = NTuple{length(it.data), AbstractArray}

# function batchselect(x::AbstractArray, i)
#     inds = CartesianIndices(size(x)[1:end-1])
#     x[inds, i]
# end

# xs = rand(10)
# ys = rand(2, 10)
# dataloader = DataLoader(xs, ys, batchsize=3, shuffle=true)
# for (x, y) in dataloader
#     @show x
# end
