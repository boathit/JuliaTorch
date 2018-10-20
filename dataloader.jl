using Random:randperm!

struct DataLoader
    data::Tuple
    batchsize::Int
    shuffle::Bool
    indices::Vector{Int}
    n::Int
end

function DataLoader(data::NTuple{N, AbstractArray}; batchsize::Int=100, shuffle=false) where N
    lens = [last(size(x)) for x in data]
    n = first(lens)
    @assert all(len -> len == n, lens) "The last dimension of all the Arrays should be equal"
    # n = last(size(first(data)))
    indices = collect(1:n)
    shuffle && randperm!(indices)
    DataLoader(data, batchsize, shuffle, indices, n)
end

"""
    DataLoader(data...; batchsize::Int=100, shuffle=true)

DataLoader provides iterators over the dataset (data...).

```julia
xs = rand(10)
ys = rand(2, 10)
dataloader = DataLoader(xs, ys, batchsize=3, shuffle=true)

for (x, y) in dataloader
    @show x
end
```
"""
DataLoader(data...; batchsize::Int=100, shuffle=false) =
    DataLoader(data, batchsize=batchsize, shuffle=shuffle)

function Base.iterate(it::DataLoader, start=1)
    if start > it.n
        it.shuffle && randperm!(it.indices)
        return nothing
    end
    nextstart = min(start + it.batchsize, it.n + 1)
    i = it.indices[start:nextstart-1]
    element = Tuple(copy(selectdim(x, ndims(x), i)) for x in it.data)
    return element, nextstart
end

Base.length(it::DataLoader) = it.n
Base.eltype(it::DataLoader) = NTuple{length(it.data), AbstractArray}

# function batchselect(x::AbstractArray, i)
#     inds = CartesianIndices(size(x)[1:end-1])
#     x[inds, i]
# end
