# Use baremodule to shave off a few KB from the serialized `.ji` file
baremodule Cbc_jll
using Base
using Base: UUID
import JLLWrappers

JLLWrappers.@generate_main_file_header("Cbc")
JLLWrappers.@generate_main_file("Cbc", UUID("38041ee0-ae04-5750-a4d2-bb4d0d83d27d"))
end  # module Cbc_jll
