
module TestPipelineStatic

using Test
using MLJBase

t  = MLJBase.WrappedFunction(f=log)
f, = fit(t, 0)
@test transform(t, f, 5) ≈ log(5)

end
true
