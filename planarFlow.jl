using PyCall

@pyimport torch
@pyimport torch.nn as nn
@pyimport torch.nn.functional as F
@pyimport torch.optim as optim
@pyimport torch.nn.utils as utils
@pyimport torch.distributions as dist


device = torch.device(ifelse(torch.cuda[:is_available](), "cuda", "cpu"))
println(device)


"""
D = 5
z = torch.rand(10, D)
m = PlanarFlow(D)
m(z)
m[:log_abs_det_jacobian](z)
"""
@pydef mutable struct PlanarFlow <: nn.Module
    function __init__(self, D)
        pybuiltin(:super)(PlanarFlow, self)[:__init__]()
        self[:w] = nn.Parameter(torch.randn(D, 1))
        self[:b] = nn.Parameter(torch.tensor(0.0))
        self[:u] = nn.Parameter(torch.randn(D, 1))

        ## such an intialization is critically important to the fast convergence
        self[:w][:data][:uniform_](-0.01, 0.01)
        self[:u][:data][:uniform_](-0.01, 0.01)
        #nn.init[:kaiming_uniform_](self[:w], a=sqrt(5))
        #nn.init[:kaiming_uniform_](self[:u], a=sqrt(5))
    end
    ## planar flow
    function forward(self, z)
       h = F.leaky_relu(torch.mm(z, self[:w]) + self[:b]) # nx1
       z + torch.t(self[:u]) * h # nxD + 1xD * nx1
    end
    function log_abs_det_jacobian(self, z)
        h = F.leaky_relu(torch.mm(z, self[:w]) + self[:b]) # nx1
        I = torch.ones_like(h)
        A = I * 0.01
        d = torch.where(h > 0, I, A) # derivative of prelu
        φ = torch.t(self[:w]) * d # 1xD * nx1 => nxD
        det = 1 + torch.mm(φ, self[:u]) # nx1
        #println(d)
        torch.squeeze(torch.log(torch.abs(det)))
    end
end



"""
K, D = 2, 5
m = PlanarFlows(K, D)
z = torch.randn(10, D)
m[:log_abs_det_jacobian](z)
"""
@pydef mutable struct PlanarFlows <: nn.Module
    function __init__(self, K, D)
        pybuiltin(:super)(PlanarFlows, self)[:__init__]()
        self[:flows] = nn.ModuleList([PlanarFlow(D) for _ in 1:K])
    end
    function forward(self, z)
        x = z
        for flow in self[:flows]
            x = flow(x)
        end
        x
    end
    function log_abs_det_jacobian(self, z)
        x, J = z, 0
        for flow in self[:flows]
            J = J + flow[:log_abs_det_jacobian](x)
            x = flow(x)
        end
        J
    end
end

## assume we have already known the target distribution P(x)
function logP(x)
    x1 = x[:index_select](1, torch.tensor([0], device=device)) |> torch.squeeze
    x2 = x[:index_select](1, torch.tensor([1], device=device)) |> torch.squeeze
    μ, σ = torch.tensor(0.0, device=device), torch.tensor(4.0, device=device)
    x2_dist = dist.Normal(μ, σ)
    x1_dist = dist.Normal(.25 * x2[:pow](2), torch.ones_like(x2))
    x2_dist[:log_prob](x2) + x1_dist[:log_prob](x1)
end

function lossF(q, z, m)
    logQ = q[:log_prob](z)
    J = m[:log_abs_det_jacobian](z)
    logQx = logQ - J # estimated log prob
    F.l1_loss(logP(m(z)), logQx)
end

n, K, D = 256, 10, 2
m = PlanarFlows(K, D)[:to](device)
optimizer = optim.Adam(m[:parameters](), lr=0.001)
μ, σ = torch.zeros(D, device=device), torch.eye(D, device=device)
q = dist.MultivariateNormal(μ, σ)

function train(nepoch)
    num_iterations = 5000
    for epoch in 1:nepoch
        epochLoss = 0
        for i in 1:num_iterations
            ## generate z
            z = q[:sample]((n,))
            loss = lossF(q, z, m)
            epochLoss += loss[:item]()

            optimizer[:zero_grad]()
            loss[:backward]()
            #utils.clip_grad_norm_(m[:parameters](), 1.0)
            optimizer[:step]()
            i % 1000 == 0 && GC.gc(false)
        end
        println("Epoch: $epoch\tLoss: $(epochLoss/num_iterations)")
    end
end

@time train(5)

using PyPlot

# ## target distribution
# x2_dist = dist.Normal(0, 4)
# x2 = x2_dist[:sample]((1000,))
# x1_dist = dist.Normal(.25 * x2[:pow](2), torch.ones_like(x2))
# x1 = x1_dist[:sample]()
# target_x = torch.stack([x1, x2], dim=1)
# plot(x1[:cpu]()[:numpy](), x2[:cpu]()[:numpy](), "r.")

## transformed distribution
x = m(q[:sample]((1000,))[:to](device))[:detach]()
x1 = x[:index_select](1, torch.tensor([0])) |> torch.squeeze
x2 = x[:index_select](1, torch.tensor([1])) |> torch.squeeze
plot(x1[:cpu]()[:numpy](), x2[:cpu]()[:numpy](), "r.")
