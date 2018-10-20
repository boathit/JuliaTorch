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
            default=10
    end
    parse_args(s; as_symbols=true)
end

for (arg, val) in args
    println("$arg => $val")
end

device = torch.device(ifelse(!args[:nocuda] && torch.cuda[:is_available](), "cuda", "cpu"))
println(device)

trainLoader, testLoader = getmnistDataLoaders(args[:batchsize])

@pydef mutable struct Encoder <: nn.Module
    function __init__(self)
        pybuiltin(:super)(Encoder, self)[:__init__]()
        self[:fc] = nn.Linear(28^2, 400)
        self[:fμ] = nn.Linear(400, 20)
        self[:flogσ] = nn.Linear(400, 20)
    end

    function forward(self, x)
        h = F.relu(self[:fc](x))
        ## μ, logσ
        self[:fμ](h), self[:flogσ](h)
    end
end

@pydef mutable struct Decoder <: nn.Module
    function __init__(self)
        pybuiltin(:super)(Decoder, self)[:__init__]()
        self[:f] = nn.Sequential(nn.Linear(20, 400),
                                 nn.ReLU(),
                                 nn.Linear(400, 784),
                                 nn.LogSigmoid())
    end

    forward(self, z) = self[:f](z)
end

function reparameterize(μ, logσ)
    σ = torch.exp(logσ)
    ϵ = torch.randn_like(σ)
    μ + σ * ϵ
end

function vae(encoder, decoder, x)
    μ, logσ = encoder(x)
    z = reparameterize(μ, logσ)
    logx̂ = decoder(z)
    logx̂, μ, logσ
end

function lossF(logx̂, x, μ, logσ)
    term1 = F.binary_cross_entropy_with_logits(logx̂, x, reduction="sum")
    term2 = -0.5 * torch.sum(1 + 2logσ - torch.pow(μ, 2) - torch.exp(2logσ))
    term1 + term2
end

function train!(encoder, decoder, optimizer_encoder, optimizer_decoder, nepoch)
    for epoch in 1:nepoch
        epochLoss = 0.0
        for (x, _) in trainLoader
            x = x[:reshape](-1, 784)[:to](device)
            logx̂, μ, logσ = vae(encoder, decoder, x)
            loss = lossF(logx̂, x, μ, logσ)
            epochLoss += loss[:item]()

            optimizer_encoder[:zero_grad]()
            optimizer_decoder[:zero_grad]()
            loss[:backward]()
            optimizer_encoder[:step]()
            optimizer_decoder[:step]()
        end
        println("Epoch: $epoch\t Loss: $(epochLoss/60_000)")
        GC.gc(false)
    end
end

function drawSamples(decoder, n)
    @pywith torch.no_grad() begin
        z = torch.randn(n, 20)[:to](device)
        x = torch.exp(decoder(z))[:cpu]()
        return x[:reshape](n, 28, 28)[:numpy]()
    end
end

encoder, decoder = Encoder()[:to](device), Decoder()[:to](device)
optimizer_encoder = optim.Adam(encoder[:parameters](), lr=0.001)
optimizer_decoder = optim.Adam(decoder[:parameters](), lr=0.001)

println("Training...")

@time train!(encoder, decoder, optimizer_encoder, optimizer_decoder, args[:nepoch])

### Testing in Jupyter notebook ###
# using PyPlot
# x = drawSamples(decoder, 20)
# plt[:imshow](x[1, :, :])
