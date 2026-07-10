cmake_minimum_required(VERSION 3.15...3.27)
project(wosx-bindings)

if(WOSX_ENABLE_GPU_SUPPORT)
    set(WOSX_BINDINGS_SRC_FILES
        "wosx_bindings.cpp"
        "wosx_bindings.h"
        "wosx_gpu_bindings.h"
    )
else()
    set(WOSX_BINDINGS_SRC_FILES
        "wosx_bindings.cpp"
        "wosx_bindings.h"
    )
endif()

nanobind_add_module(_wosx NOMINSIZE STABLE_ABI NB_STATIC ${WOSX_BINDINGS_SRC_FILES})
target_link_libraries(_wosx PRIVATE wosx)

# ensure the extension can find adjacent shared libs at runtime
if(UNIX AND NOT APPLE)
    # use $ORIGIN so the module can load libs placed next to it after install
    set_target_properties(_wosx PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
    # use @loader_path on macOS
    set_target_properties(_wosx PROPERTIES BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
endif()

install(TARGETS _wosx
    LIBRARY DESTINATION wosx
    RUNTIME DESTINATION wosx
)

# Bundle the TBB runtime used by the extension. The vendored oneTBB
# install rules are disabled globally, so install the shared library next
# to _wosx where Python can find it on Windows and via RPATH elsewhere.
if(TARGET tbb)
    install(TARGETS tbb
        LIBRARY DESTINATION wosx
        RUNTIME DESTINATION wosx
    )
endif()

# install GPU support files if enabled
if(WOSX_ENABLE_GPU_SUPPORT)
    # Package the slang shaders, which GPU task handles compile at runtime.
    # The source-tree relative layout is preserved under wosx/slang so that
    # GPULibraryPaths resolves modules from an installed package without any
    # changes; pass wosx.slang_directory_path as the wosx_directory_path argument
    # of GPUTaskHandle when using an installed package rather than a source checkout.
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/wosx/gpu
        DESTINATION wosx/slang/include/wosx
        FILES_MATCHING PATTERN "*.slang"
    )
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/deps/fcpw/include/fcpw/gpu
        DESTINATION wosx/slang/deps/fcpw/include/fcpw
        FILES_MATCHING PATTERN "*.slang"
    )

    # slang-rhi is static by default, but it depends on fetched Slang
    # compiler/runtime libraries and optional backend DLLs. Bundle those
    # next to _wosx so GPU bindings do not depend on build-tree paths.
    if(CMAKE_CONFIGURATION_TYPES)
        set(WOSX_SLANG_RHI_RUNTIME_DIR "${CMAKE_BINARY_DIR}/\${CMAKE_INSTALL_CONFIG_NAME}")
    else()
        set(WOSX_SLANG_RHI_RUNTIME_DIR "${CMAKE_BINARY_DIR}")
    endif()

    if(WIN32)
        # slang-rhi copies Slang, DXC, WinPix, Dawn, and similar runtime DLLs
        # into its runtime output directory via the slang-rhi-copy-files target.
        install(CODE "
            set(_wosx_slang_rhi_runtime_dir \"${WOSX_SLANG_RHI_RUNTIME_DIR}\")
            file(GLOB _wosx_slang_rhi_libs \"\${_wosx_slang_rhi_runtime_dir}/*.dll\")
            foreach(_lib \${_wosx_slang_rhi_libs})
                message(STATUS \"Installing Slang RHI runtime library: \${_lib}\")
                file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/wosx\" TYPE SHARED_LIBRARY FILES \"\${_lib}\")
            endforeach()
            if(EXISTS \"\${_wosx_slang_rhi_runtime_dir}/D3D12\")
                message(STATUS \"Installing Slang RHI D3D12 runtime directory: \${_wosx_slang_rhi_runtime_dir}/D3D12\")
                file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/wosx\" TYPE DIRECTORY FILES \"\${_wosx_slang_rhi_runtime_dir}/D3D12\")
            endif()
        ")
    else()
        if(DEFINED SLANG_RHI_SLANG_BINARY_DIR)
            if(APPLE)
                install(CODE "
                    file(GLOB _wosx_slang_libs
                        \"${SLANG_RHI_SLANG_BINARY_DIR}/lib/libslang*.dylib\"
                        \"${SLANG_RHI_SLANG_BINARY_DIR}/lib/libgfx*.dylib\")
                    foreach(_lib \${_wosx_slang_libs})
                        message(STATUS \"Installing Slang library: \${_lib}\")
                        file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/wosx\" TYPE SHARED_LIBRARY FILES \"\${_lib}\")
                    endforeach()
                ")
            else()
                install(CODE "
                    file(GLOB _wosx_slang_libs
                        \"${SLANG_RHI_SLANG_BINARY_DIR}/lib/libslang*.so*\"
                        \"${SLANG_RHI_SLANG_BINARY_DIR}/lib/libgfx*.so*\")
                    foreach(_lib \${_wosx_slang_libs})
                        message(STATUS \"Installing Slang library: \${_lib}\")
                        file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/wosx\" TYPE SHARED_LIBRARY FILES \"\${_lib}\")
                    endforeach()
                ")
            endif()
        else()
            message(WARNING "WOSX_ENABLE_GPU_SUPPORT is ON but SLANG_RHI_SLANG_BINARY_DIR is not defined. "
                            "GPU functionality may not work in Python bindings.")
        endif()

        # install optional copied backend runtimes such as Dawn if present.
        install(CODE "
            set(_wosx_slang_rhi_runtime_dir \"${WOSX_SLANG_RHI_RUNTIME_DIR}\")
            file(GLOB _wosx_slang_rhi_libs
                \"\${_wosx_slang_rhi_runtime_dir}/*.so*\"
                \"\${_wosx_slang_rhi_runtime_dir}/*.dylib\")
            foreach(_lib \${_wosx_slang_rhi_libs})
                message(STATUS \"Installing Slang RHI runtime library: \${_lib}\")
                file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/wosx\" TYPE SHARED_LIBRARY FILES \"\${_lib}\")
            endforeach()
        ")
    endif()
endif()

if(MSVC)
    target_compile_options(_wosx PRIVATE /bigobj)
endif()
