# Autogenerated wrapper script for Cairo_jll for x86_64-w64-mingw32
export libcairo, libcairo_gobject, libcairo_script_interpreter

using Glib_jll
using Pixman_jll
using libpng_jll
using Fontconfig_jll
using FreeType2_jll
using Bzip2_jll
using Xorg_libXext_jll
using Xorg_libXrender_jll
using LZO_jll
using Zlib_jll
JLLWrappers.@generate_wrapper_header("Cairo")
JLLWrappers.@declare_library_product(libcairo, "libcairo-2.dll")
JLLWrappers.@declare_library_product(libcairo_gobject, "libcairo-gobject-2.dll")
JLLWrappers.@declare_library_product(libcairo_script_interpreter, "libcairo-script-interpreter-2.dll")
function __init__()
    JLLWrappers.@generate_init_header(Glib_jll, Pixman_jll, libpng_jll, Fontconfig_jll, FreeType2_jll, Bzip2_jll, Xorg_libXext_jll, Xorg_libXrender_jll, LZO_jll, Zlib_jll)
    JLLWrappers.@init_library_product(
        libcairo,
        "bin\\libcairo-2.dll",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libcairo_gobject,
        "bin\\libcairo-gobject-2.dll",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libcairo_script_interpreter,
        "bin\\libcairo-script-interpreter-2.dll",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
