
using PyCall

torch = pyimport("torch")
optim = pyimport("torch.optim")
nn    = pyimport("torch.nn")
F     = pyimport("torch.nn.functional")

## Generating data

x = torch.rand(1000, 100)
w = torch.rand(100, 1)
y = x.matmul(w)

## Defining model and optimizer
model = nn.Linear(100, 1, bias=false)
optimizer = optim.Adam(model.parameters(), lr=0.01)

## Optimizing model
for i = 0:5000
    ŷ = model(x)
    loss = F.mse_loss(ŷ, y)
    i % 500 == 0 && println("Loss is $(loss.item())")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
end
