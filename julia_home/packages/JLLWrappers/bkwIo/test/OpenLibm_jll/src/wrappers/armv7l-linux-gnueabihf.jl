# Autogenerated wrapper script for OpenLibm_jll for armv7l-linux-gnueabihf
export libopenlibm

JLLWrappers.@generate_wrapper_header("OpenLibm")
JLLWrappers.@declare_library_product(libopenlibm, "libopenlibm.so.3")
JLLWrappers.@declare_library_product(libnonexisting, "libnonexisting.so.0")
function __init__()
    JLLWrappers.@generate_init_header()
    JLLWrappers.@init_library_product(
        libopenlibm,
        "lib/libopenlibm.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )
    JLLWrappers.@init_library_product(
        libnonexisting,
        "lib/libnonexisting.so",
        nothing,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
