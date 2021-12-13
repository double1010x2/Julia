# Use baremodule to shave off a few KB from the serialized `.ji` file
baremodule CoinUtils_jll
using Base
using Base: UUID
import JLLWrappers

JLLWrappers.@generate_main_file_header("CoinUtils")
JLLWrappers.@generate_main_file("CoinUtils", UUID("be027038-0da8-5614-b30d-e42594cb92df"))
end  # module CoinUtils_jll
