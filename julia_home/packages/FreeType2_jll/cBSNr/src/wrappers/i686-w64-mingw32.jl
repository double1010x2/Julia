# Autogenerated wrapper script for FreeType2_jll for i686-w64-mingw32
export libfreetype

using Bzip2_jll
using Zlib_jll
JLLWrappers.@generate_wrapper_header("FreeType2")
JLLWrappers.@declare_library_product(libfreetype, "libfreetype-6.dll")
function __init__()
    JLLWrappers.@generate_init_header(Bzip2_jll, Zlib_jll)
    JLLWrappers.@init_library_product(
        libfreetype,
        "bin\\libfreetype-6.dll",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
