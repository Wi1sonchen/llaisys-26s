local corex_home = os.getenv("COREX_HOME") or "/usr/local/corex-4.4.0"
local corex_bin = os.getenv("COREX_BIN") or "/usr/local/corex/bin"
local corex_lib = os.getenv("COREX_LIB") or "/usr/local/corex/lib64"

local corex_cxx = path.join(corex_bin, "clang++")

local function setup_corex_target()
    set_languages("cxx17")
    set_warnings("all", "error")

    set_toolset("cxx", corex_cxx)

    add_includedirs(
        path.join(corex_home, "include")
    )

    add_linkdirs(
        corex_lib,
        {public = true}
    )
end


target("llaisys-device-nvidia")
    set_kind("static")

    setup_corex_target()

    add_files(
        "../src/device/nvidia/*.cu",
        {
            sourcekind = "cxx",

            force = {
                cxxflags = {
                    "-x",
                    "ivcore",
                    "--cuda-gpu-arch=ivcore11",
                    "--cuda-path=" .. corex_home,
                    "-fPIC"
                }
            }
        }
    )

    add_links(
        "cudart",
        {public = true}
    )

    on_install(function (target)
    end)
target_end()


target("llaisys-ops-nvidia")
    set_kind("static")

    setup_corex_target()

    add_files(
        "../src/ops/*/nvidia/*.cu",
        {
            sourcekind = "cxx",

            force = {
                cxxflags = {
                    "-x",
                    "ivcore",
                    "--cuda-gpu-arch=ivcore11",
                    "--cuda-path=" .. corex_home,
                    "-fPIC"
                }
            }
        }
    )

    add_links(
        "cudart",
        {public = true}
    )

    add_links(
        "cublas",
        {public = true}
    )

    on_install(function (target)
    end)
target_end()
