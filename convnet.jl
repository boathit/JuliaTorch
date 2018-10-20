using PyCall
using ArgParse
include("dataUtils.jl")

@pyimport torch
@pyimport torch.nn as nn
@pyimport torch.nn.functional as F
@pyimport torch.optim as optim

args = let s = ArgParseSettings()
    @add_arg_table s begin
        "--nocuda"
            action=:store_true
        "--batchsize"
            arg_type=Int
            default=256
        "--nepoch"
            arg_type=Int
            default=20
    end
    parse_args(s; as_symbols=true)
end

for (arg, val) in args
    println("$arg => $val")
end

device = torch.device(ifelse(!args[:nocuda] && torch.cuda[:is_available](), "cuda", "cpu"))
println(device)

trainLoader, testLoader = getmnistDataLoaders(args[:batchsize])

@pydef mutable struct ConvNet <: nn.Module
    function __init__(self)
        pybuiltin(:super)(ConvNet, self)[:__init__]()
        self[:layer1] = nn.Sequential(nn.Conv2d(1, 20, kernel_size=5),
                                      nn.MaxPool2d(2),
                                      nn.ReLU(),
                                      nn.Conv2d(20, 50, kernel_size=5),
                                      nn.MaxPool2d(2),
                                      nn.ReLU())
        self[:layer2] = nn.Sequential(nn.Linear(800, 500),
                                      nn.ReLU(),
                                      nn.Linear(500, 10),
                                      nn.LogSoftmax(dim=1))
    end

    function forward(self, x)
        h = self[:layer1](x)
        o = self[:layer2](h[:reshape](-1, 800))
    end
end

model = ConvNet()[:to](device)
optimizer = optim.Adam(model[:parameters](), lr=0.001)

function train()
    for epoch in 1:args[:nepoch]
        for (i, (x, y)) in enumerate(trainLoader)
            (x, y) = x[:to](device), y[:to](device)
            o = model(x)
            loss = F.nll_loss(o, y)

            optimizer[:zero_grad]()
            loss[:backward]()
            optimizer[:step]()
            i % 100 == 0 && println("Epoch: $epoch\tLoss: $(loss[:item]())")
        end
        GC.gc(false)
    end
end

println("Training...")

@time train()

println("Testing...")

let (n, N) = (0, 0)
    @pywith torch.no_grad() begin
        for (x, y) in testLoader
            (x, y) = x[:to](device), y[:to](device)
            o = model(x)
            _, ŷ = torch.max(o, 1)
            N += y[:size](0)
            n += torch.sum(ŷ == y)[:item]()
        end
        GC.gc(false)
        println("Accuracy: $(n/N)")
    end
end
