module TestArrows

using MLJBase
using ..Models
using Test
using Random
using StableRNGs

@testset "|> syntax for pipelines" begin
    rng = StableRNG(56161)
    X = MLJBase.table(randn(rng, 500, 5))
    y = abs.(randn(rng, 500))
    train, test = partition(eachindex(y), 0.7)

    # Feeding data directly to a supervised model
    knn = KNNRegressor(K=10)
    ŷ   = (X, y) |> knn
    fit!(ŷ, rows=train, verbosity=0)

    # Describing a full pipeline using |> syntax.
    Xs, ys = source.((X, y))

    # "first layer"
    W = Xs |> Standardizer()
    z = ys |> UnivariateBoxCoxTransformer()
    # "second layer"
    ẑ = (W, z) |> RidgeRegressor(lambda=0.1)
    # "output layer"
    ŷ = ẑ |> inverse_transform(z)

    fit!(ŷ, rows=train, verbosity=0)

    @test isapprox(rms(ŷ(rows=test), ys(rows=test)), 0.59, atol=1e-2)

    # shortcut to get and set hyperparameters of a node
    ẑ[:lambda] = 5.0
    fit!(ŷ, rows=train, verbosity=0)
    @test isapprox(rms(ŷ(rows=test), ys(rows=test)), 0.59, atol=1e-2)

end

@testset "Auto-source" begin
    rng = StableRNG(51616161)

    X = MLJBase.table(randn(rng,500, 5))
    y = abs.(randn(rng,500))

    W = X |> Standardizer() |> PCA(maxoutdim=2)
    fit!(W, verbosity=0)

    Wraw = W()
    sch = schema(Wraw)
    @test sch.names == (:x1, :x2)
    @test sch.scitypes == (MLJBase.Continuous, MLJBase.Continuous)
    @test sch.nrows == 500

    yhat = (W, y) |> RidgeRegressor()
    fit!(yhat, verbosity=0)

    yhat_raw = yhat()
    @test yhat_raw isa Vector{Float64}
    @test length(yhat_raw) == 500
end

@testset "Auto-table" begin
    rng = StableRNG(5661)

    X = randn(rng,500, 5)
    y = abs.(randn(rng,500))

    W = X |> Standardizer() |> PCA(maxoutdim=2)
    yhat = (W, y) |> RidgeRegressor()
    fit!(yhat, verbosity=0)

    yhat_raw = yhat()
    @test yhat_raw isa Vector{Float64}
    @test length(yhat_raw) == 500
end

@testset "Stacking" begin
    rng = StableRNG(66161)

    X = randn(rng,500, 5)
    y = abs.(randn(rng,500))

    W = X |> Standardizer() |> PCA(maxoutdim=3)
    z = y |> UnivariateBoxCoxTransformer()
    ẑ₁ = (W, z) |> RidgeRegressor()
    ẑ₂ = (W, z) |> DecisionTreeRegressor()
    R = hcat(ẑ₁, ẑ₂)
    ẑ = (R, z) |> DecisionTreeRegressor()
    ŷ = ẑ |> inverse_transform(z)

    fit!(ŷ, verbosity=0)

    p̂ = ŷ()
    @test p̂ isa Vector{Float64}
    @test length(p̂) == 500
end

@testset "functions and static transfomers" begin
    rng = StableRNG(66666)
    x1 = rand(rng,30)
    x2 = rand(rng,30)
    x3 = rand(rng,30)
    yy = exp.(x1 - x2 -2x3 + 0.1*rand(rng,30))
    XX = (x1=x1, x2=x2, x3=x3)

    f(X) = (a=selectcols(X, :x1), b=selectcols(X, :x2))

    knn = KNNRegressor()

    XXs = source(XX)
    yys = source(yy)

    f(X::AbstractNode) = node(f, X)
    WW = XXs |> f |> Standardizer()
    z = yys |> UnivariateBoxCoxTransformer()
    zhat = (WW, z) |> knn
    yyhat = zhat |> inverse_transform(z)
    fit!(yyhat, verbosity=0)
    pred = yyhat();

    mutable struct MyTransformer <: Static
        ftr::Symbol
    end
    MLJBase.transform(transf::MyTransformer, verbosity, XX) =
        (a=selectcols(XX, transf.ftr), b=selectcols(XX, :x2))
    inserter = MyTransformer(:x1)
    WW = XXs |> inserter |> Standardizer()
    z = yys |> UnivariateBoxCoxTransformer()
    zhat = (WW, z) |> knn
    yyhat = zhat |> inverse_transform(z)
    fit!(yyhat, verbosity=0)
    @test yyhat() ≈ pred
end

end
true
