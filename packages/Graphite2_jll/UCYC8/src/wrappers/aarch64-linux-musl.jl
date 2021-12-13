# Autogenerated wrapper script for Graphite2_jll for aarch64-linux-musl
export libgraphite2

JLLWrappers.@generate_wrapper_header("Graphite2")
JLLWrappers.@declare_library_product(libgraphite2, "libgraphite2.so.3")
function __init__()
    JLLWrappers.@generate_init_header()
    JLLWrappers.@init_library_product(
        libgraphite2,
        "lib/libgraphite2.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
