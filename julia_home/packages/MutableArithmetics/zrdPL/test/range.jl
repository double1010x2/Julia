using Test
import MutableArithmetics
const MA = MutableArithmetics

function mutating_step_range_test(::Type{T}) where T
    r = MA.MutatingStepRange(T(2), T(3), T(9))
    expected = MA.mutability(T) isa MA.IsMutable ? 8 * ones(T, 3) : T[2, 5, 8]
    @test collect(r) == expected
    @test reduce(MA.add!!, r) == T(15)
end

@testset "MutatingStepRange" begin
    mutating_step_range_test(Int)
    mutating_step_range_test(BigInt)
end
