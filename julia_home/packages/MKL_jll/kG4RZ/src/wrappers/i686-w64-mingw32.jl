# Autogenerated wrapper script for MKL_jll for i686-w64-mingw32
export libmkl_core, libmkl_rt

using IntelOpenMP_jll
JLLWrappers.@generate_wrapper_header("MKL")
JLLWrappers.@declare_library_product(libmkl_core, "mkl_core.1.dll")
JLLWrappers.@declare_library_product(libmkl_rt, "mkl_rt.1.dll")
function __init__()
    JLLWrappers.@generate_init_header(IntelOpenMP_jll)
    JLLWrappers.@init_library_product(
        libmkl_core,
        "bin\\mkl_core.1.dll",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libmkl_rt,
        "bin\\mkl_rt.1.dll",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()