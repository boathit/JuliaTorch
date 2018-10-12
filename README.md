# JuliaTorch

Using PyTorch in Julia Language via PyCall.

## Requirement

* [Julia](http://julialang.org) 1.0+
* [PyTorch](http://pytorch.org) 0.4+ with Python 3 (Anaconda3 is recommended)
* [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) 1.18.5+

You can tell PyCall.jl which Python you would like to use by exporting `PYTHON` environment variable in your `.bashrc`
or `.zshrc`, e.g., if I wish to use the Python in Anaconda3 I can add the line

```shell
export PYTHON="/home/xiucheng/anaconda3/bin/python"
```

## Roadmap

* [Linear Regression](https://github.com/boathit/JuliaTorch/blob/master/linearRegression.jl)
* [Logistic Regression](https://github.com/boathit/JuliaTorch/blob/master/logisticRegression.jl)
* [MLP](https://github.com/boathit/JuliaTorch/blob/master/mlp.jl)
* [Convolutional Neural Net](https://github.com/boathit/JuliaTorch/blob/master/convnet.jl)
* [Residual Neural Net](https://github.com/boathit/JuliaTorch/blob/master/resnet.jl), using skip connection:
  `julia resnet.jl`; using plain connection: `julia resnet.jl --plain`.
* [Variational Auto-Encoder](https://github.com/boathit/JuliaTorch/blob/master/vae.jl), implementing
  VAE described in [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).

## Usage

The codes can be run in command line or Jupyter notebook. For example,

```shell
$ julia vae.jl --nepoch 15
```

## Miscell

Defining PyTorch nn.Module in Julia

```julia
@pydef mutable struct ResBlock <: nn.Module
    function __init__(self, ...)
        pybuiltin(:super)(ResBlock, self)[:__init__]()
        self[:f] = ...
    end
    function forward(self, x)
      self[:f](x)
    end
end
```
