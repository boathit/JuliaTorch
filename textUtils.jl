## pip install -U spacy
## python -m spacy download en
## add StatsBase in julia

using PyCall
using StatsBase

@pyimport spacy

nlp = spacy.load("en", disable=["ner", "tagger", "parser"])

function cleantext(s::String)
    txt = replace(s, r"[^A-Za-z0-9.?!;]" => " ") # remove extra characters
    txt = replace(txt, r" +"=>" ") # remove extra spaces
    txt = split(txt, " ") .|> lowercase |> strs -> join(strs, " ") # lowercase
    doc = nlp(txt)

    tokens = String[]
    for token in doc
        !token[:is_stop] && push!(tokens, token[:lemma_])
    end
    join(tokens, " ")
end

# doc = nlp("account doings summon")
# for token in doc
#     println(token[:text], "|", token[:lemma_], "|", token[:is_stop])
# end

function buildVocab(words::Vector{String})
    unigram = countmap(words)
    pairs = sort(collect(unigram), by=last, rev=true)
    word2idx = Dict([word=>i-1 for (i, word) in enumerate(first.(pairs))])
    idx2word = Dict([last(p)=>first(p) for p in word2idx])
    probs = last.(pairs)
    word2idx, idx2word, collect(0:length(probs)-1), probs ./ sum(probs)
end

#word2idx, idx2word, vocab, probs = buildVocab(words)

function buildVocab(text::String)
    words = filter(w -> length(w) > 1, split(text, ['.', '?', '!', ';', ' ']))
    words = String.(words)
    buildVocab(words)
end

function createContext(wordidx::Vector{Int}, contextsize::Int)
    δ = div(contextsize, 2)
    w = Int[]
    C = Vector{Int}[]
    for i = 1+δ:length(wordidx)-δ
        push!(w, wordidx[i])
        push!(C, vcat(wordidx[i-δ:i-1], wordidx[i+1:i+δ]))
    end
    w, hcat(C...)
end

function createContext(word2idx::Dict, text::String, contextsize::Int)
    sentences = String.(split(text, ['.', '?', '!', ';']))
    function sentence2idx(sentence)
        words = filter(w -> length(w) > 1, split(sentence, ' '))
        words = String.(words)
        wordidx = map(x -> get(word2idx, x, 0), words)
    end
    wordidxs = filter(s -> length(s) > contextsize, sentence2idx.(sentences))
    wordidxs = convert(Vector{Vector{Int}}, wordidxs)
    wC = createContext.(wordidxs, contextsize)
    vcat(first.(wC)...), hcat(last.(wC)...)
end

# s = read("../data/sherlock-holmes.txt", String);
# text = cleantext(s);
#
# word2idx, idx2word, vocab, probs = buildVocab(text)
#
# w, C = createContext(word2idx, text, 2);
