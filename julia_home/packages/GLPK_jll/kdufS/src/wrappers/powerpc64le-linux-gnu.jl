# Autogenerated wrapper script for GLPK_jll for powerpc64le-linux-gnu
export libglpk

using GMP_jll
JLLWrappers.@generate_wrapper_header("GLPK")
JLLWrappers.@declare_library_product(libglpk, "libglpk.so.40")
function __init__()
    JLLWrappers.@generate_init_header(GMP_jll)
    JLLWrappers.@init_library_product(
        libglpk,
        "lib/libglpk.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
