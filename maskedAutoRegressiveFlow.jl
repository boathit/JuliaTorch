using PyCall
using PyPlot

include("autoRegressiveNN.jl")

torch = pyimport("torch")
optim = pyimport("torch.optim")
nn    = pyimport("torch.nn")
F     = pyimport("torch.nn.functional")
dist  = pyimport("torch.distributions")


device = torch.device(ifelse(torch.cuda.is_available(), "cuda", "cpu"))
println(device)

clampx(x, min, max) = x + (x.clamp(min, max) - x).detach()

"""
n, D = 10, 5
maf = MAF(AutoRegressiveNN(D, [8], 2))
u = torch.rand(n, D)
x = maf._call(u)
maf._inverse(x) - u
maf.log_abs_det_jacobian(u, x)

pybuiltin(:list)(maf.m.parameters())

mafs = [MAF(AutoRegressiveNN(D, [2D], 2)), MAF(AutoRegressiveNN(D, [2D], 2))]
u = torch.rand(n, D)
foward(u) = foldl((t, maf) -> maf._call(t), mafs; init=u)
inverse(x) = foldl((t, maf) -> maf._inverse(t), reverse(mafs); init=x)
x = foward(u)
inverse(x) - u
"""
@pydef mutable struct MAF <: dist.Transform
    function __init__(self, arn)
        pybuiltin(:super)(MAF, self).__init__()
        self.D = D
        ## self.m holds all the parameters
        self.m = nn.Module()
        self.m.arn = arn
        self.order = self.m.arn.order
    end
    ## u (n, D) -> x (n, D)
    function _call(self, u)
        x = torch.zeros_like(u)
        n = u.shape[1]
        for i in self.order
            index = torch.tensor(i-1)
            μ, α = self.m.arn(x)
            μi = μ.index_select(1, index)
            αi = α.index_select(1, index)
            αi = clampx(αi, -5.0, 3.0)

            ui = u.index_select(1, index)
            xi = μi + ui * torch.exp(αi)
            x.index_copy_(1, index, xi)
        end
        x
    end
    ## x (n, D) -> u (n, D)
    function _inverse(self, x)
        μ, α = self.m.arn(x)
        α = clampx(α, -5.0, 3.0)
        (x - μ) * torch.exp(-α)
    end
    function log_abs_det_jacobian(self, u, x)
        _, α = self.m.arn(x)
        α = clampx(α, -5.0, 3.0)
        #return α.sum(dim=1, keepdim=true)
    end
end

## target distribution
function drawP(n)
    x2_dist = dist.Normal(0, 4)
    x2 = x2_dist.sample((n,))
    x1_dist = dist.Normal(.25 * x2.pow(2), torch.ones_like(x2))
    x1 = x1_dist.sample()
    #plot(x1.cpu().numpy(), x2.cpu().numpy(), "r.")
    torch.stack([x1, x2], dim=1).to(device)
end
