# Autogenerated wrapper script for IntelOpenMP_jll for x86_64-apple-darwin
export libiomp

JLLWrappers.@generate_wrapper_header("IntelOpenMP")
JLLWrappers.@declare_library_product(libiomp, "libiomp5.dylib")
function __init__()
    JLLWrappers.@generate_init_header()
    JLLWrappers.@init_library_product(
        libiomp,
        "lib/libiomp5.dylib",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
