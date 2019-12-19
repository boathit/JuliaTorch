## pip install -U spacy
## python -m spacy download en
## add StatsBase in julia

using PyCall
using StatsBase

spacy = pyimport("spacy")

nlp = spacy.load("en", disable=["ner", "tagger", "parser"])

function cleantext(s::String)
    txt = replace(s, r"[^A-Za-z0-9.?!;]" => " ") # remove extra characters
    txt = replace(txt, r" +"=>" ") # remove extra spaces
    txt = split(txt, " ") .|> lowercase |> strs -> join(strs, " ") # lowercase
    doc = nlp(txt)

    tokens = String[]
    for token in doc
        !token.is_stop && push!(tokens, token.lemma_)
    end
    join(tokens, " ")
end

# doc = nlp("account doings summon")
# for token in doc
#     println(token[:text], "|", token[:lemma_], "|", token[:is_stop])
# end

"""
Split a text into multiple sentences; each sentence is a sequence of words.
"""
function text2seqwords(text::String)
    sentences = String.(split(text, ['.', '?', '!', ';']))
    function sentence2words(sentence)
        words = filter(w -> length(w) > 1, split(sentence, ' '))
        String.(words)
    end
    sentence2words.(sentences)
end

function buildVocab(words::Vector{String})
    unigram = countmap(words)
    pairs = sort(collect(unigram), by=last, rev=true)
    word2idx = Dict([word=>i-1 for (i, word) in enumerate(first.(pairs))])
    idx2word = Dict([last(p)=>first(p) for p in word2idx])
    probs = last.(pairs)
    word2idx, idx2word, collect(0:length(probs)-1), probs ./ sum(probs)
end

#word2idx, idx2word, vocab, probs = buildVocab(words)

buildVocab(seqwords::Vector{Vector{String}}) = buildVocab(vcat(seqwords...))

"""
Creating context for a sequence of tokens.
"""
function createContext(tokens::Vector{Int}, contextsize::Int)
    @assert iseven(contextsize) "context size should be a even."
    δ, n = div(contextsize, 2), length(tokens)
    w, C = Int[], Vector{Int}[]
    for i = 1:n
        push!(w, tokens[i])
        if i <= δ
            push!(C, vcat(tokens[1:i-1], tokens[i+1:2δ+1]))
        elseif i > n - δ
            push!(C, vcat(tokens[n-2δ:i-1], tokens[i+1:n]))
        else # [1+δ, n-δ]
            push!(C, vcat(tokens[i-δ:i-1], tokens[i+1:i+δ]))
        end
    end
    w, hcat(C...)
end


function createContext(seqwords::Vector{Vector{String}},
                       word2idx::Dict, contextsize::Int)
    words2idx(words) = map(x -> get(word2idx, x, 0), words)
    seqtokens = filter(s -> length(s) > contextsize, words2idx.(seqwords))
    seqtokens = convert(Vector{Vector{Int}}, seqtokens)
    wC = createContext.(seqtokens, contextsize)
    vcat(first.(wC)...), hcat(last.(wC)...)
end


# s = read("../data/sherlock-holmes.txt", String);
# text = cleantext(s);
# seqwords = text2seqwords(text)
#
# word2idx, idx2word, vocab, probs = buildVocab(seqwords);
#
# w, C = createContext(seqwords, word2idx, 4);
