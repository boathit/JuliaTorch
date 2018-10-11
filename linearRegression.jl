
using PyCall

@pyimport torch
@pyimport torch.nn as nn
@pyimport torch.nn.functional as F
@pyimport torch.optim as optim

@pydef mutable struct Net <: nn.Module
    function __init__(self)
        pybuiltin(:super)(Net, self)[:__init__]()
        self[:fc1] = nn.Linear(100, 1, bias=false)
    end

    function forward(self, x)
        h1 = self[:fc1](x)
    end
end

## Generating data

x = torch.rand(1000, 100)
w = torch.rand(100, 1)
y = x[:matmul](w)

## Defining model and optimizer
net = Net()
optimizer = optim.Adam(net[:parameters](), lr=0.01)

## Optimizing model
for i = 0:5000
    ŷ = net(x)
    loss = F.mse_loss(ŷ, y)
    i % 500 == 0 && println("Loss is $(loss[:item]())")
    optimizer[:zero_grad]()
    loss[:backward]()
    optimizer[:step]()
end
