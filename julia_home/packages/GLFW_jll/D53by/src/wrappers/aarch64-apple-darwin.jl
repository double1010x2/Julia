# Autogenerated wrapper script for GLFW_jll for aarch64-apple-darwin
export libglfw

using Libglvnd_jll
using Xorg_libXcursor_jll
using Xorg_libXi_jll
using Xorg_libXinerama_jll
using Xorg_libXrandr_jll
JLLWrappers.@generate_wrapper_header("GLFW")
JLLWrappers.@declare_library_product(libglfw, "@rpath/libglfw.3.dylib")
function __init__()
    JLLWrappers.@generate_init_header(Libglvnd_jll, Xorg_libXcursor_jll, Xorg_libXi_jll, Xorg_libXinerama_jll, Xorg_libXrandr_jll)
    JLLWrappers.@init_library_product(
        libglfw,
        "lib/libglfw.3.3.dylib",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
