using SafeTestsets

@safetestset "diagrams" begin
    include("diagrams.jl")
end
@safetestset "aqua" begin
    include("aqua.jl")
end
