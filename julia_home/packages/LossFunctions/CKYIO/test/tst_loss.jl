function test_value_typestable(l::SupervisedLoss)
    @testset "$(l): " begin
        for y in (-1, 1, Int32(-1), Int32(1), -1.5, 1.5, Float32(-.5), Float32(.5))
            for t in (-2, 2, Int32(-1), Int32(1), -.5, .5, Float32(-1), Float32(1))
                # check inference
                @inferred deriv(l, y, t)
                @inferred deriv2(l, y, t)

                # get expected return type
                T = promote_type(typeof(y), typeof(t))

                # test basic loss
                val = @inferred value(l, y, t)
                @test typeof(val) <: T

                # test scaled version of loss
                @test typeof(value(T(2)*l, y, t)) <: T
            end
        end
    end
end

function test_value_float32_preserving(l::SupervisedLoss)
    @testset "$(l): " begin
        for y in (-1, 1, Int32(-1), Int32(1), -1.5, 1.5, Float32(-.5), Float32(.5))
            for t in (-2, 2, Int32(-1), Int32(1), -.5, .5, Float32(-1), Float32(1))
                # check inference
                @inferred deriv(l, y, t)
                @inferred deriv2(l, y, t)

                val = @inferred value(l, y, t)
                T = promote_type(typeof(y),typeof(t))
                if !(T <: AbstractFloat)
                    # cast Integers to a float
                    # (whether its Float32 or Float64 depends on the loss...)
                    @test (typeof(val) <: AbstractFloat)
                elseif T <: Float32
                    # preserve Float32
                    @test (typeof(val) <: Float32)
                else
                    @test (typeof(val) <: Float64)
                end
            end
        end
    end
end

function test_value_float64_forcing(l::SupervisedLoss)
    @testset "$(l): " begin
        for y in (-1, 1, Int32(-1), Int32(1), -1.5, 1.5, Float32(-.5), Float32(.5))
            for t in (-2, 2, Int32(-1), Int32(1), -.5, .5, Float32(-1), Float32(1))
                # check inference
                @inferred deriv(l, y, t)
                @inferred deriv2(l, y, t)

                val = @inferred value(l, y, t)
                @test (typeof(val) <: Float64)
            end
        end
    end
end

function test_value(l::SupervisedLoss, f::Function, y_vec, t_vec)
    @testset "$(l): " begin
        for y in y_vec, t in t_vec
            @test abs(value(l, y, t) - f(y, t)) < 1e-10
        end
    end
end

function test_deriv(l::MarginLoss, t_vec)
    @testset "$(l): " begin
        for y in [-1., 1], t in t_vec
            if isdifferentiable(l, y*t)
                d_dual = epsilon(value(l, dual(y, zero(y)), dual(t, one(t))))
                d_comp = @inferred deriv(l, y, t)
                @test abs(d_dual - d_comp) < 1e-10
                val = @inferred value(l, y, t)
                @test val ≈ value(l, y, t)
                @test val ≈ value(l, y*t)
                @test d_comp ≈ y*deriv(l, y*t)
            else
                # y*t == 1 ? print(".") : print("(no $(y)*$(t)) ")
                #print(".")
            end
        end
    end
end

function test_deriv(l::DistanceLoss, t_vec)
    @testset "$(l): " begin
        for y in -10:.2:10, t in t_vec
            if isdifferentiable(l, t-y)
                d_dual = epsilon(value(l, dual(t-y, one(t-y))))
                d_comp = @inferred deriv(l, y, t)
                @test abs(d_dual - d_comp) < 1e-10
                val = @inferred value(l, y, t)
                @test val ≈ value(l, y, t)
                @test val ≈ value(l, t-y)
                @test d_comp ≈ deriv(l, t-y)
            end
        end
    end
end

function test_deriv(l::SupervisedLoss, y_vec, t_vec)
    @testset "$(l): " begin
        for y in y_vec, t in t_vec
            if isdifferentiable(l, y, t)
                d_dual = epsilon(value(l, y, dual(t, one(t))))
                d_comp = @inferred deriv(l, y, t)
                @test abs(d_dual - d_comp) < 1e-10
                val = @inferred value(l, y, t)
                @test val ≈ value(l, y, t)
                @test d_comp ≈ deriv(l, y, t)
            end
        end
    end
end

function test_deriv2(l::MarginLoss, t_vec)
    @testset "$(l): " begin
        for y in [-1., 1], t in t_vec
            if istwicedifferentiable(l, y*t) && isdifferentiable(l, y*t)
                d2_dual = epsilon(deriv(l, dual(y, zero(y)), dual(t, one(t))))
                d2_comp = @inferred deriv2(l, y, t)
                @test abs(d2_dual - d2_comp) < 1e-10
                @test d2_comp ≈ @inferred deriv2(l, y, t)
                @test d2_comp ≈ @inferred deriv2(l, y*t)
            end
        end
    end
end

function test_deriv2(l::DistanceLoss, t_vec)
    @testset "$(l): " begin
        for y in -10:.2:10, t in t_vec
            if istwicedifferentiable(l, t-y) && isdifferentiable(l, t-y)
                d2_dual = epsilon(deriv(l, dual(t-y, one(t-y))))
                d2_comp = @inferred deriv2(l, y, t)
                @test abs(d2_dual - d2_comp) < 1e-10
                @test d2_comp ≈ @inferred deriv2(l, y, t)
                @test d2_comp ≈ @inferred deriv2(l, t-y)
            end
        end
    end
end

function test_deriv2(l::SupervisedLoss, y_vec, t_vec)
    @testset "$(l): " begin
        for y in y_vec, t in t_vec
            if istwicedifferentiable(l, y, t) && isdifferentiable(l, y, t)
                d2_dual = epsilon(deriv(l, dual(y, zero(y)), dual(t, one(t))))
                d2_comp = @inferred deriv2(l, y, t)
                @test abs(d2_dual - d2_comp) < 1e-10
                @test d2_comp ≈ @inferred deriv2(l, y, t)
            end
        end
    end
end

function test_scaledloss(l::SupervisedLoss, t_vec, y_vec)
    @testset "Scaling for $(l): " begin
        for λ = (2.0, 2)
            sl = ScaledLoss(l,λ)
            @test typeof(sl) <: ScaledLoss{typeof(l),λ}
            @test 3 * sl == @inferred(ScaledLoss(sl,Val(3)))
            @test (λ*3) * l == @inferred(ScaledLoss(sl,Val(3)))
            @test sl == @inferred(ScaledLoss(l,Val(λ)))
            @test sl == λ * l
            @test sl == @inferred(Val(λ) * l)
            for t in t_vec
                for y in y_vec
                    @test @inferred(value(sl,t,y)) == λ*value(l,t,y)
                    @test @inferred(deriv(sl,t,y)) == λ*deriv(l,t,y)
                    @test @inferred(deriv2(sl,t,y)) == λ*deriv2(l,t,y)
                end
            end
        end
    end
end

function test_scaledloss(l::SupervisedLoss, n_vec)
    @testset "Scaling for $(l): " begin
        for λ = (2.0, 2)
            sl = ScaledLoss(l,λ)
            @test typeof(sl) <: ScaledLoss{typeof(l),λ}
            @test sl == @inferred(ScaledLoss(l,Val(λ)))
            @test sl == λ * l
            @test sl == @inferred(Val(λ) * l)
        end
    end
end

function test_weightedloss(l::MarginLoss, t_vec, y_vec)
    @testset "Weighted version for $(l): " begin
        for w in (0., 0.2, 0.7, 1.)
            wl = WeightedMarginLoss(l, w)
            @test typeof(wl) <: WeightedMarginLoss{typeof(l),w}
            @test WeightedMarginLoss(l, w * 0.1) == WeightedMarginLoss(wl, 0.1)
            for t in t_vec
                for y in y_vec
                    if t == 1
                        @test value(wl,t,y) == w*value(l,t,y)
                        @test deriv(wl,t,y) == w*deriv(l,t,y)
                        @test deriv2(wl,t,y) == w*deriv2(l,t,y)
                    else
                        @test value(wl,t,y) == (1-w)*value(l,t,y)
                        @test deriv(wl,t,y) == (1-w)*deriv(l,t,y)
                        @test deriv2(wl,t,y) == (1-w)*deriv2(l,t,y)
                    end
                end
            end
        end
    end
end

# ====================================================================

@testset "Test typealias" begin
    @test L1DistLoss === LPDistLoss{1}
    @test L2DistLoss === LPDistLoss{2}
    @test HingeLoss === L1HingeLoss
    @test EpsilonInsLoss === L1EpsilonInsLoss
end

@testset "Test typestable supervised loss for type stability" begin
    for loss in [L1HingeLoss(), L2HingeLoss(), ModifiedHuberLoss(),
                 PerceptronLoss(), LPDistLoss(1), LPDistLoss(2),
                 LPDistLoss(3), L2MarginLoss()]
        test_value_typestable(loss)
        # TODO: add ZeroOneLoss after scaling works...
    end
end

@testset "Test float-forcing supervised loss for type stability" begin
    # Losses that should always return Float64
    for loss in [SmoothedL1HingeLoss(0.5), SmoothedL1HingeLoss(1),
                 L1EpsilonInsLoss(0.5), L1EpsilonInsLoss(1),
                 L2EpsilonInsLoss(0.5), L2EpsilonInsLoss(1),
                 PeriodicLoss(1), PeriodicLoss(1.5),
                 HuberLoss(1.0), QuantileLoss(.8),
                 DWDMarginLoss(0.5), DWDMarginLoss(1), DWDMarginLoss(2)]
        test_value_float64_forcing(loss)
        test_value_float64_forcing(2.0 * loss)
    end
    test_value_float64_forcing(2.0 * LogitDistLoss())
    test_value_float64_forcing(2.0 * LogitMarginLoss())
    test_value_float64_forcing(2.0 * ExpLoss())
    test_value_float64_forcing(2.0 * SigmoidLoss())

    # Losses that should return an AbstractFloat, preserving type if possible
    for loss in [SmoothedL1HingeLoss(0.5f0), SmoothedL1HingeLoss(1f0),
                 PeriodicLoss(1f0), PeriodicLoss(0.5f0),
                 LogitDistLoss(), LogitMarginLoss(), ExpLoss(), SigmoidLoss(),
                 L1EpsilonInsLoss(1f0), L1EpsilonInsLoss(0.5f0),
                 L2EpsilonInsLoss(1f0), L2EpsilonInsLoss(0.5f0),
                 HuberLoss(1.0f0), QuantileLoss(.8f0), DWDMarginLoss(0.5f0)]
        test_value_float32_preserving(loss)
        test_value_float32_preserving(2f0 * loss)
    end
end

println("<HEARTBEAT>")

@testset "Test margin-based loss against reference function" begin
    _zerooneloss(y, t) = sign(y*t) < 0 ? 1 : 0
    test_value(ZeroOneLoss(), _zerooneloss, [-1.,1], -10:0.2:10)

    _hingeloss(y, t) = max(0, 1 - y.*t)
    test_value(HingeLoss(), _hingeloss, [-1.,1], -10:0.2:10)

    _l2hingeloss(y, t) = max(0, 1 - y.*t)^2
    test_value(L2HingeLoss(), _l2hingeloss, [-1.,1], -10:0.2:10)

    _perceptronloss(y, t) = max(0, -y.*t)
    test_value(PerceptronLoss(), _perceptronloss, [-1.,1], -10:0.2:10)

    _logitmarginloss(y, t) = log(1 + exp(-y.*t))
    test_value(LogitMarginLoss(), _logitmarginloss, [-1.,1], -10:0.2:10)

    function _smoothedl1hingeloss(γ)
        function _value(y, t)
            if y.*t >= 1 - γ
                1/(2γ) * max(0, 1- y.*t)^2
            else
                1 - γ / 2 - y.*t
            end
        end
        _value
    end
    test_value(SmoothedL1HingeLoss(.5), _smoothedl1hingeloss(.5), [-1.,1], -10:0.2:10)
    test_value(SmoothedL1HingeLoss(1), _smoothedl1hingeloss(1), [-1.,1], -10:0.2:10)
    test_value(SmoothedL1HingeLoss(2), _smoothedl1hingeloss(2), [-1.,1], -10:0.2:10)

    function _modhuberloss(y, t)
        if y .* t >= -1
            max(0, 1 - y .* t)^2
        else
            -4 .* y .* t
        end
    end
    test_value(ModifiedHuberLoss(), _modhuberloss, [-1.,1], -10:0.2:10)

    _l2marginloss(y, t) = (1 - y.*t)^2
    test_value(L2MarginLoss(), _l2marginloss, [-1.,1], -10:0.2:10)

    _exploss(y, t) = exp(-y.*t)
    test_value(ExpLoss(), _exploss, [-1.,1], -10:0.2:10)

    _sigmoidloss(y, t) = (1-tanh(y.*t))
    test_value(SigmoidLoss(), _sigmoidloss, [-1., 1], -10:0.2:10)

    function _dwdmarginloss(q)
        function _value(y, t)
            if y.*t <= q/(q+1)
                convert(Float64, 1 - y.*t)
            else
                ((q^q)/(q+1)^(q+1)) / (y.*t)^q
            end
        end
        _value
    end
    test_value(DWDMarginLoss(0.5), _dwdmarginloss(0.5), [-1., 1], -10:0.2:10)
    test_value(DWDMarginLoss(1), _dwdmarginloss(1), [-1., 1], -10:0.2:10)
    test_value(DWDMarginLoss(2), _dwdmarginloss(2), [-1., 1], -10:0.2:10)

end

@testset "Test distance-based loss against reference function" begin
    yr, tr = range(-10, stop=20, length=10), range(-30, stop=30, length=10)

    _l1distloss(y, t) = abs(t - y)
    test_value(L1DistLoss(), _l1distloss, yr, tr)

    _l2distloss(y, t) = (t - y)^2
    test_value(L2DistLoss(), _l2distloss, yr, tr)

    _lp15distloss(y, t) = abs(t - y)^(1.5)
    test_value(LPDistLoss(1.5), _lp15distloss, yr, tr)

    function _periodicloss(c)
        _value(y, t) = 1 - cos((y-t)*2π/c)
        _value
    end
    test_value(PeriodicLoss(0.5), _periodicloss(0.5), yr, tr)
    test_value(PeriodicLoss(1), _periodicloss(1), yr, tr)
    test_value(PeriodicLoss(1.5), _periodicloss(1.5), yr, tr)

    function _huberloss(d)
        _value(y, t) = abs(y-t)<d ? (abs2(y-t)/2) : (d*(abs(y-t) - (d/2)))
        _value
    end
    test_value(HuberLoss(0.5), _huberloss(0.5), yr, tr)
    test_value(HuberLoss(1), _huberloss(1), yr, tr)
    test_value(HuberLoss(1.5), _huberloss(1.5), yr, tr)

    function _l1epsinsloss(ɛ)
        _value(y, t) = max(0, abs(t - y) - ɛ)
        _value
    end
    test_value(EpsilonInsLoss(0.5), _l1epsinsloss(0.5), yr, tr)
    test_value(EpsilonInsLoss(1), _l1epsinsloss(1), yr, tr)
    test_value(EpsilonInsLoss(1.5), _l1epsinsloss(1.5), yr, tr)

    function _l2epsinsloss(ɛ)
        _value(y, t) = max(0, abs(t - y) - ɛ)^2
        _value
    end
    test_value(L2EpsilonInsLoss(0.5), _l2epsinsloss(0.5), yr, tr)
    test_value(L2EpsilonInsLoss(1), _l2epsinsloss(1), yr, tr)
    test_value(L2EpsilonInsLoss(1.5), _l2epsinsloss(1.5), yr, tr)

    _logitdistloss(y, t) = -log((4*exp(t-y))/(1+exp(t-y))^2)
    test_value(LogitDistLoss(), _logitdistloss, yr, tr)

    function _quantileloss(y, t)
        (y - t) * (0.7 - (y - t < 0))
    end
    test_value(QuantileLoss(.7), _quantileloss, yr, tr)

    function _logcoshloss(y, t)
        log.(cosh(y .- t))
    end
    test_value(LogCoshLoss(), _logcoshloss, yr, tr)
end

const OrdinalSmoothedHingeLoss = OrdinalMarginLoss{<:SmoothedL1HingeLoss}
@test OrdinalSmoothedHingeLoss(4, 2.1) === OrdinalMarginLoss(SmoothedL1HingeLoss(2.1), 4)

@testset "Test ordinal losses against reference function" begin
    function _ordinalhingeloss(y, t)
        val = 0
        for yp = 1:y - 1
            val += max(0, 1 - t + yp)
        end
        for yp = y + 1:5
            val += max(0, 1 + t - yp)
        end
        val
    end
    y = rand(1:5, 10); t = randn(10) .+ 3
    test_value(OrdinalMarginLoss(HingeLoss(), 5), _ordinalhingeloss, y, t)
end


@testset "Test other loss against reference function" begin
    _misclassloss(y, t) = y == t ? 0 : 1
    test_value(MisclassLoss(), _misclassloss, 1:10, vcat(1:5,7:11))

    _crossentropyloss(y, t) = -y*log(t) - (1-y)*log(1-t)
    test_value(CrossEntropyLoss(), _crossentropyloss, 0:0.01:1, 0.01:0.01:0.99)

    _poissonloss(y, t) = exp(t) - t*y
    test_value(PoissonLoss(), _poissonloss, 0:10, range(0,stop=10,length=11))
    test_scaledloss(PoissonLoss(), 0:10, range(0,stop=10,length=11))
end

println("<HEARTBEAT>")

# --------------------------------------------------------------

@testset "Test first derivatives of margin-based losses" begin
    for loss in margin_losses
        test_deriv(loss, -10:0.2:10)
    end
end

@testset "Test second derivatives of margin-based losses" begin
    for loss in margin_losses
        test_deriv2(loss, -10:0.2:10)
    end
end

@testset "Test margin-based scaled loss" begin
    for loss in margin_losses
        test_scaledloss(loss, [-1.,1], -10:0.2:10)
        test_scaledloss(loss, -10:0.2:10)
    end
end

@testset "Test margin-based weighted loss" begin
    for loss in margin_losses
        test_weightedloss(loss, [-1.,1], -10:0.2:10)
    end
end

# --------------------------------------------------------------

@testset "Test first derivatives of distance-based losses" begin
    for loss in distance_losses
        test_deriv(loss, -10:0.5:10)
    end
end

@testset "Test second derivatives of distance-based losses" begin
    for loss in distance_losses
        test_deriv2(loss, -10:0.5:10)
    end
end

@testset "Test first and second derivatives of other losses" begin
    test_deriv(PoissonLoss(), -10:.2:10, 0:30)
    test_deriv2(PoissonLoss(), -10:.2:10, 0:30)
    test_deriv(CrossEntropyLoss(), 0:0.01:1, 0.01:0.01:0.99)
    test_deriv2(CrossEntropyLoss(), 0:0.01:1, 0.01:0.01:0.99)
end

@testset "Test distance-based scaled loss" begin
    for loss in distance_losses
        test_scaledloss(loss, -10:.2:10, -10:0.5:10)
        test_scaledloss(loss, -10:0.5:10)
    end
end

# --------------------------------------------------------------

@testset "Losses with categorical values" begin
    c = categorical(["Foo","Bar","Baz","Foo"])

    l = MisclassLoss()
    @test value(l, c[1], c[1]) == 0.0
    @test value(l, c[1], c[2]) == 1.0
    @test value(l, c, reverse(c)) == [0.0, 1.0, 1.0, 0.0]
    @test value(l, c, reverse(c), AggMode.Sum()) == 2.0
    @test value(l, c, reverse(c), AggMode.Mean()) == 0.5
    @test value(l, c, reverse(c), AggMode.WeightedSum(2*ones(4))) == 4.0
    @test value(l, c, reverse(c), AggMode.WeightedMean(2*ones(4),false)) == 1.0
    @test value(l, c, reverse(c), AggMode.WeightedMean(2*ones(4),true)) == 0.125

    lf = MisclassLoss{Float32}()
    @test value(lf, c[1], c[1]) isa Float32
    @test value(lf, c, c) isa Vector{Float32}
end
