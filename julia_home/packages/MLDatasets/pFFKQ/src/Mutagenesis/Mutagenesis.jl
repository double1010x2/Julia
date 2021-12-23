export Mutagenesis

"""
Mutagenesis

Website: https://relational.fit.cvut.cz/dataset/Mutagenesis
License: CC0

The `Mutagenesis` dataset comprises 188 molecules trialed for mutagenicity on Salmonella typhimurium, available from
 [relational.fit.cvut.cz](https://relational.fit.cvut.cz/dataset/Mutagenesis) and
 [CTUAvastLab/datasets](https://github.com/CTUAvastLab/datasets/tree/main/mutagenesis).

Train, test and validation data can be loaded using following code.
The `withenv("DATADEPS_ALWAYS_ACCEPT"=>"true")` disables the accept download prompt.
```jldoctest
julia> using MLDatasets: Mutagenesis

julia> train_x, train_y = withenv("DATADEPS_ALWAYS_ACCEPT"=>"true") do; Mutagenesis.traindata(); end;

julia> test_x, test_y = Mutagenesis.testdata();

julia> val_x, val_y = Mutagenesis.valdata();

julia> train_x[1]
Dict{Symbol, Any} with 5 entries:
  :lumo  => -1.246
  :inda  => 0
  :logp  => 4.23
  :ind1  => 1
  :atoms => Dict{Symbol, Any}[Dict(:element=>"c", :bonds=>Dict{Symbol, Any}[Dic…

julia> train_y[1]
1
```
"""
module Mutagenesis

using DataDeps, JSON3
using ..MLDatasets: datafile

const DEPNAME = "Mutagenesis"
const DATA = "data.json"
const METADATA = "meta.json"

function __init__()
    ORIGINAL_LINK = "https://relational.fit.cvut.cz/dataset/Mutagenesis"
    DATA_LINK = "https://raw.githubusercontent.com/CTUAvastLab/datasets/main/mutagenesis"

    register(DataDep(
        DEPNAME,
        """
        Dataset: The $DEPNAME dataset.
        Website: $ORIGINAL_LINK
        License: CC0
        """,
        "$DATA_LINK/" .* [DATA, METADATA],
        "80ec1716217135e1f2e0b5a61876c65184e2014e64551103c41e174775ca207c"
    ))
end

traindata(; dir = nothing) = traindata(dir)
testdata(; dir = nothing) = testdata(dir)
valdata(; dir = nothing) = valdata(dir)

function traindata(dir)
    samples, targets, train_idxs, val_idxs, test_idxs = load_data(dir)
    samples[train_idxs], targets[train_idxs]
end

function testdata(dir)
    samples, targets, train_idxs, val_idxs, test_idxs = load_data(dir)
    samples[test_idxs], targets[test_idxs]
end

function valdata(dir)
    samples, targets, train_idxs, val_idxs, test_idxs = load_data(dir)
    samples[val_idxs], targets[val_idxs]
end

function load_data(dir)
    data_path = datafile(DEPNAME, DATA, dir)
    metadata_path = datafile(DEPNAME, METADATA, dir)
    samples = read_data(data_path)
    metadata = read_data(metadata_path)
    labelkey = metadata["label"]
    targets = map(i -> i[labelkey], samples)
    samples_without_label = map(x->delete!(copy(x), Symbol(labelkey)), samples)
    val_num = metadata["val_samples"]
    test_num = metadata["test_samples"]
    train_idxs = 1:length(samples)-val_num-test_num
    val_idxs = length(samples)-val_num-test_num+1:length(samples)-test_num
    test_idxs = length(samples)-test_num+1:length(samples)
    samples_without_label, targets, train_idxs, val_idxs, test_idxs
end

read_data(path) = open(JSON3.read, path)

end # module
