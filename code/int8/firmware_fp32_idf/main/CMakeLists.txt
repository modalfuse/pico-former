idf_component_register(SRCS app_main.cpp
                       INCLUDE_DIRS ../../../pico-former/firmware/include
                       REQUIRES esp_driver_uart esp_timer heap)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-unused-variable -Wno-unused-function)
