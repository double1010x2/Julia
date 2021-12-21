# Autogenerated wrapper script for Cbc_jll for x86_64-w64-mingw32-libgfortran5-cxx03
export cbc, libCbc, libOsiCbc, libcbcsolver

using ASL_jll
using Cgl_jll
using Clp_jll
using Osi_jll
using CoinUtils_jll
using OpenBLAS32_jll
using CompilerSupportLibraries_jll
JLLWrappers.@generate_wrapper_header("Cbc")
JLLWrappers.@declare_library_product(libCbc, "libCbc-3.dll")
JLLWrappers.@declare_library_product(libOsiCbc, "libOsiCbc-3.dll")
JLLWrappers.@declare_library_product(libcbcsolver, "libCbcSolver-3.dll")
JLLWrappers.@declare_executable_product(cbc)
function __init__()
    JLLWrappers.@generate_init_header(ASL_jll, Cgl_jll, Clp_jll, Osi_jll, CoinUtils_jll, OpenBLAS32_jll, CompilerSupportLibraries_jll)
    JLLWrappers.@init_library_product(
        libCbc,
        "bin\\libCbc-3.dll",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libOsiCbc,
        "bin\\libOsiCbc-3.dll",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libcbcsolver,
        "bin\\libCbcSolver-3.dll",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_executable_product(
        cbc,
        "bin\\cbc.exe",
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()