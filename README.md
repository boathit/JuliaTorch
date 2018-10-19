# JuliaTorch

Using PyTorch in [Julia Language](http://julialang.org) via [PyCall](https://github.com/JuliaPy/PyCall.jl).

## Requirement

* Julia >= 1.0
* PyTorch >= 0.4 with Python 3 (Anaconda3 is recommended)
* PyCall >= 1.18.5

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
* [Residual Neural Net](https://github.com/boathit/JuliaTorch/blob/master/resnet.jl) implements
  resnet described in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
* [Word2Vec](https://github.com/boathit/JuliaTorch/blob/master/word2vec.jl) implements Skip-Gram described in [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546).
* [Variational Auto-Encoder](https://github.com/boathit/JuliaTorch/blob/master/vae.jl) implements
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
