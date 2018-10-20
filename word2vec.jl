## [paper](https://arxiv.org/abs/1310.4546)

using PyCall
using ArgParse
using StatsBase

@pyimport torch
@pyimport torch.nn as nn
@pyimport torch.nn.functional as F
@pyimport torch.optim as optim

include("dataloader.jl")
include("textUtils.jl")

args = let s = ArgParseSettings()
    @add_arg_table s begin
        "--nocuda"
            action=:store_true
        "--batchsize"
            arg_type=Int
            default=128
        "--nepoch"
            arg_type=Int
            default=50
    end
    parse_args(s; as_symbols=true)
end

for (arg, val) in args
    println("$arg => $val")
end

device = torch.device(ifelse(!args[:nocuda] && torch.cuda[:is_available](), "cuda", "cpu"))
println(device)

## parameter settings
batchsize = args[:batchsize]
embdim = 100
contextsize = 4
negativesamplesize = 20

@pydef mutable struct Word2Vec <: nn.Module
    function __init__(self, vocabsize, embdim, contextsize)
        pybuiltin(:super)(Word2Vec, self)[:__init__]()
        ## input vector representations (see [paper])
        self[:U] = nn.Embedding(vocabsize, embdim)
        ## output vector representations (see [paper])
        self[:V] = nn.Embedding(vocabsize, embdim)
        ## context window size or the number of positive samples
        self[:m] = contextsize
    end

    ## here we use the same negative samples for every positive sample of w[i],
    ## thus we times `m`
    forward(self, w, P, N) = self[:positiveLoss](w, P) + self[:m] * self[:negativeLoss](w, N)
    positiveLoss(self, w, P) = self[:sampleLoss](w, P, true)
    negativeLoss(self, w, N) = self[:sampleLoss](w, N, false)

    function sampleLoss(self, w, X, positive=true)
        """
        Return the loss defined by the positive or negative samples.

        Input:
        w (n): the center word ids, where n is the batchsize.
        X (n, m|k): positive|negative sample ids.
        """
        u = self[:U](w)[:unsqueeze_](2) # (n, embdim, 1)
        v = self[:V](X) # (n, m|k, embdim)
        ## the inner product of v and u, i.e., <v, u>
        x = torch.bmm(v, u)[:squeeze_](2) # (n, m|k)
        positive ? -torch.mean(F.logsigmoid(x)) : -torch.mean(F.logsigmoid(-x))
    end
end

## loading data
text = read("./data/sherlock-holmes.txt", String)
text = cleantext(text)
seqwords = text2seqwords(text)
word2idx, idx2word, vocab, probs = buildVocab(seqwords)
centerwords, context = createContext(seqwords, word2idx, contextsize)
probs .= probs .^ (3/4)
probs .= probs ./ sum(probs)

# i = 4
# map(x->get(idx2word, x, "UNK"), context[:, i])
# get(idx2word, centerwords[i], "UNK")

model = Word2Vec(length(vocab), embdim, contextsize)[:to](device)
optimizer = optim.Adam(model[:parameters](), lr=0.001)
dataloader = DataLoader(centerwords, context, batchsize=batchsize, shuffle=true)

## Drawing `k` negative samples, here we do not exclude the center word and positive samples
function negativeSamples(vocab, probs, n, k)
    ## (k, n)
    hcat([sample(vocab, Weights(probs), k, replace=false) for _ in 1:n]...)
end

function train!(model, optimizer, dataloader, vocab, probs, nepoch)
    numstep = length(dataloader) / batchsize
    for epoch in 1:nepoch
        epochloss = 0.0
        for (i, (w, P)) in enumerate(dataloader)
            ## w (n) is the center word ids, P (n, m) is the positive sample ids
            P = P' |> copy
            N = negativeSamples(vocab, probs, length(w), negativesamplesize)' |> copy
            loss = model(torch.LongTensor(w)[:to](device),
                         torch.LongTensor(P)[:to](device),
                         torch.LongTensor(N)[:to](device))
            epochloss += loss[:item]()

            optimizer[:zero_grad]()
            loss[:backward]()
            optimizer[:step]()
            # i % 10 == 0 && GC.gc(false)
        end
        GC.gc(false)
        println("Epoch: $epoch\t Loss: $(epochloss/numstep)")
    end
end

println("There are $(length(dataloader)) data points.\nTraining...")
@time train!(model, optimizer, dataloader, vocab, probs, args[:nepoch])

## saving word embeddings and idx2word
U = model[:U][:weight][:data][:cpu]()[:numpy]()' |> copy
V = model[:V][:weight][:data][:cpu]()[:numpy]()' |> copy
using BSON: @save
@save "w2vec.bson" U V idx2word word2idx

## playing around in REPL
using BSON: @load
using Distances
@load "w2vec.bson" U V idx2word word2idx

function knn(word2idx::Dict, idx2word::Dict, U::Matrix, word::String, k::Int=10)
    idx = get(word2idx, word, -1) + 1
    idx == 0 && (println("cannot find the word: $word"); return nothing)
    dists = pairwise(Euclidean(), U[:, idx:idx], U)[:]
    ord = sort([(i-1, x) for (i, x) in enumerate(dists)], by=last) .|> first
    map(i -> get(idx2word, i, "UNK"), ord[2:k+1])
end

@show knn(word2idx, idx2word, U, "door")
@show knn(word2idx, idx2word, U, "see")
@show knn(word2idx, idx2word, U, "kill")
