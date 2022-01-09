
@testset "Deprecations" begin
    # RodriguesVec
    @test Base.isdeprecated(Rotations, :UnitQuaternion)
end
