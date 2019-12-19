using PyCall
using ArgParse
include("dataUtils.jl")

torch = pyimport("torch")
optim = pyimport("torch.optim")
nn    = pyimport("torch.nn")
F     = pyimport("torch.nn.functional")

args = let s = ArgParseSettings()
    @add_arg_table s begin
        "--nocuda"
            action=:store_true
        "--plain"
            action=:store_true
        "--batchsize"
            arg_type=Int
            default=128
        "--nepoch"
            arg_type=Int
            default=100
    end
    parse_args(s; as_symbols=true)
end

for (arg, val) in args
    println("$arg => $val")
end

device = torch.device(ifelse(!args[:nocuda] && torch.cuda.is_available(), "cuda", "cpu"))
println(device)

trainLoader, testLoader = getcifar10DataLoaders(args[:batchsize])

conv3x3(num_in::Int, num_out::Int, stride::Int) =
    nn.Conv2d(num_in, num_out, kernel_size=3,
              stride=stride, padding=1, bias=false)

function align(num_in::Int, num_out::Int, stride::Int)
    """
    Returning a function `f` that aligns the input and output tensors in
    residual block, i.e., f(x) will have the same shape with transforms(x)
    defined in residual block.
    """
    if  num_in != num_out || stride > 1
        nn.Sequential(conv3x3(num_in, num_out, stride),
                      nn.BatchNorm2d(num_out))
    else
        identity
    end
end

@pydef mutable struct ResBlock <: nn.Module
    function __init__(self, num_in, num_out, stride, short_cut=true)
        pybuiltin(:super)(ResBlock, self).__init__()
        self.short_cut = short_cut
        self.align = align(num_in, num_out, stride)
        ## transforms in residual block
        ## only self[:conv1] may change the shape of the input
        ## that's why we define an align function above
        self.conv1 = conv3x3(num_in, num_out, stride)
        self.bn1 = nn.BatchNorm2d(num_out)
        self.relu = nn.ReLU(inplace=true)
        self.conv2 = conv3x3(num_out, num_out, 1)
        self.bn2 = nn.BatchNorm2d(num_out)
    end
    function forward(self, x)
        ## Note that o will always have the same shape with self[:align](x)
        o = x |> self.conv1 |> self.bn1 |> self.relu |>
                 self.conv2 |> self.bn2
        self.short_cut == true && (o += self.align(x))
        self.relu(o)
    end
end

function buildResBlocks(num_in::Int, num_out::Int, stride::Int,
                        num_blocks::Int, short_cut=true)
    ## only the first block may change the shape of the input
    blocks = [ResBlock(num_in, num_out, stride, short_cut)]
    for _ in 2:num_blocks
        push!(blocks, ResBlock(num_out, num_out, 1, short_cut))
    end
    nn.Sequential(blocks...)
end

buildResBlocks(inout::Pair, stride::Int, num_blocks::Int, short_cut=true) =
    buildResBlocks(first(inout), last(inout), stride, num_blocks, short_cut)

@pydef mutable struct ResNet <: nn.Module
    function __init__(self, num_classes, short_cut=true)
        pybuiltin(:super)(ResNet, self).__init__()
        self.blocks0 = nn.Sequential(conv3x3(3, 16, 1), nn.BatchNorm2d(16),
                                     nn.ReLU(inplace=true))
        self.blocks1 = buildResBlocks(16=>16, 1, 2, short_cut)
        self.blocks2 = buildResBlocks(16=>32, 2, 2, short_cut)
        self.blocks3 = buildResBlocks(32=>64, 2, 2, short_cut)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
    end
    function forward(self, x)
        n = x.shape[1]
        x |> self.blocks0 |> self.blocks1 |> self.blocks2 |>
             self.blocks3 |> self.avgpool |> o -> o.reshape(n, -1) |>
             self.fc
    end
end

resnet = ResNet(10, !args[:plain]).to(device)
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

function adjust_lr!(optimizer, lr)
    for param in optimizer.param_groups
        param["lr"] = lr
    end
end

function train!(resnet, optimizer, nepoch)
    numstep = 60_000 / args[:batchsize]
    for epoch in 1:nepoch
        epochLoss = 0.0
        for (i, (x, y)) in enumerate(trainLoader)
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(resnet(x), y)
            epochLoss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i % 10 == 0 && GC.gc(false)
        end
        GC.gc(false)
        println("Epoch: $epoch\t Loss: $(epochLoss/numstep)")
    end
end

println("Training...")

@time train!(resnet, optimizer, args[:nepoch])

println("Testing...")
resnet.eval()
let (n, N) = (0, 0)
    @pywith torch.no_grad() begin
        for (i, (x, y)) in enumerate(testLoader)
            (x, y) = x.to(device), y.to(device)
            o = resnet(x)
            _, ŷ = torch.max(o, 1)
            N += y.size(0)
            n += torch.sum(ŷ == y).item()
            i % 10 == 0 && GC.gc(false)
        end
        GC.gc(false)
        println("Accuracy: $(n/N)")
    end
end
