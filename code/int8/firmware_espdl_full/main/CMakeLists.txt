set(srcs app_main.cpp)

idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)
set(model_file model.espdl)
if(DEFINED ENV{CEQT_ESPDL})
    set(model_file $ENV{CEQT_ESPDL})
endif()
set(embed_files ${CMAKE_CURRENT_LIST_DIR}/models/s3/${model_file})

idf_component_register(SRCS ${srcs}
                       REQUIRES esp-dl esp_driver_uart esp_timer)

target_add_aligned_binary_data(${COMPONENT_LIB} ${embed_files} BINARY)
