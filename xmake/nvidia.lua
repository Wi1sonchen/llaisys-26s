target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")

    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC")
    end

    add_files("../src/device/nvidia/*.cu")

    add_links("cudart", {public = true})

    add_values("cuda.build.devlink", true)

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")

    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC")
    end

    add_files("../src/ops/*/nvidia/*.cu")

    add_links("cudart", {public = true})
    add_links("cublas", {public = true})

    add_values("cuda.build.devlink", true)

    on_install(function (target) end)
target_end()