# Autogenerated wrapper script for Blosc_jll for armv7l-linux-gnueabihf-cxx11
export libblosc

using Zlib_jll
using Zstd_jll
using Lz4_jll
JLLWrappers.@generate_wrapper_header("Blosc")
JLLWrappers.@declare_library_product(libblosc, "libblosc.so.1")
function __init__()
    JLLWrappers.@generate_init_header(Zlib_jll, Zstd_jll, Lz4_jll)
    JLLWrappers.@init_library_product(
        libblosc,
        "lib/libblosc.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
