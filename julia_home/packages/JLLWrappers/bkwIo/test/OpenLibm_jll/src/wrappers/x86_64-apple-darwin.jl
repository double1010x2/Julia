# Autogenerated wrapper script for OpenLibm_jll for x86_64-apple-darwin14
export libopenlibm

JLLWrappers.@generate_wrapper_header("OpenLibm")
JLLWrappers.@declare_library_product(libopenlibm, "@rpath/libopenlibm.3.dylib")
JLLWrappers.@declare_library_product(libnonexisting, "libnonexisting.0.dylib")
function __init__()
    JLLWrappers.@generate_init_header()
    JLLWrappers.@init_library_product(
        libopenlibm,
        "lib/libopenlibm.3.0.dylib",
        RTLD_LAZY | RTLD_DEEPBIND,
    )
    JLLWrappers.@init_library_product(
        libnonexisting,
        "lib/libnonexisting.0.0.dylib",
        nothing,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
