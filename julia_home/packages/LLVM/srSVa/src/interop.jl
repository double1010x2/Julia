module Interop

using ..LLVM
import ..LLVM: API


const jlctx = Ref{LLVM.Context}()

include("interop/base.jl")
include("interop/asmcall.jl")
include("interop/passes.jl")
if VERSION >= v"1.5-"
include("interop/pointer.jl")
end
include("interop/utils.jl")
include("interop/intrinsics.jl")

end
