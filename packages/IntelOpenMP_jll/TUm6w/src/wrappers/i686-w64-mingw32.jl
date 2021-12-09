# Autogenerated wrapper script for IntelOpenMP_jll for i686-w64-mingw32
export libiomp

JLLWrappers.@generate_wrapper_header("IntelOpenMP")
JLLWrappers.@declare_library_product(libiomp, "libiomp5md.dll")
function __init__()
    JLLWrappers.@generate_init_header()
    JLLWrappers.@init_library_product(
        libiomp,
        "bin\\libiomp5md.dll",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
