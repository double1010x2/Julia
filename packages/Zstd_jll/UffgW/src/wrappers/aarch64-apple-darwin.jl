# Autogenerated wrapper script for Zstd_jll for aarch64-apple-darwin
export libzstd, zstd, zstdmt

JLLWrappers.@generate_wrapper_header("Zstd")
JLLWrappers.@declare_library_product(libzstd, "@rpath/libzstd.1.dylib")
JLLWrappers.@declare_executable_product(zstd)
JLLWrappers.@declare_executable_product(zstdmt)
function __init__()
    JLLWrappers.@generate_init_header()
    JLLWrappers.@init_library_product(
        libzstd,
        "lib/libzstd.1.dylib",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_executable_product(
        zstd,
        "bin/zstd",
    )

    JLLWrappers.@init_executable_product(
        zstdmt,
        "bin/zstdmt",
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
