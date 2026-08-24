include(FindPackageHandleStandardArgs)

set(ParMETIS_ROOT "" CACHE PATH "Root directory of a ParMETIS installation or source build")

set(_parmetis_triplet "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
if(ParMETIS_ROOT)
  set(_parmetis_no_default_path NO_DEFAULT_PATH)
else()
  set(_parmetis_no_default_path)
endif()

find_path(ParMETIS_INCLUDE_DIR
  NAMES parmetis.h
  HINTS "${ParMETIS_ROOT}"
  PATH_SUFFIXES include
  ${_parmetis_no_default_path})

find_path(METIS_INCLUDE_DIR
  NAMES metis.h
  HINTS "${ParMETIS_ROOT}"
  PATH_SUFFIXES include metis/include
  ${_parmetis_no_default_path})

find_library(ParMETIS_LIBRARY
  NAMES parmetis
  HINTS "${ParMETIS_ROOT}"
  PATH_SUFFIXES
    lib
    lib64
    libparmetis
    "build/${_parmetis_triplet}/libparmetis"
  ${_parmetis_no_default_path})

find_library(METIS_LIBRARY
  NAMES metis
  HINTS "${ParMETIS_ROOT}"
  PATH_SUFFIXES
    lib
    lib64
    libmetis
    "build/${_parmetis_triplet}/libmetis"
  ${_parmetis_no_default_path})

unset(ParMETIS_ABI_COMPATIBLE CACHE)
unset(ParMETIS_ABI_COMPATIBLE)
if(ParMETIS_INCLUDE_DIR AND METIS_INCLUDE_DIR)
  include(CheckCXXSourceCompiles)
  set(_parmetis_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
  set(CMAKE_REQUIRED_INCLUDES
    "${ParMETIS_INCLUDE_DIR};${METIS_INCLUDE_DIR}")
  check_cxx_source_compiles([=[
    #include <metis.h>
    #include <type_traits>
    static_assert(sizeof(long int) == 8,
                  "ParGeMSLR requires a 64-bit long int");
    static_assert(std::is_same<idx_t, long int>::value,
                  "METIS idx_t must match long int");
    static_assert(std::is_same<real_t, double>::value,
                  "METIS real_t must match double");
    int main() { return 0; }
  ]=] ParMETIS_ABI_COMPATIBLE)
  set(CMAKE_REQUIRED_INCLUDES "${_parmetis_saved_required_includes}")
endif()

find_package_handle_standard_args(ParMETIS
  REQUIRED_VARS
    ParMETIS_LIBRARY
    METIS_LIBRARY
    ParMETIS_INCLUDE_DIR
    METIS_INCLUDE_DIR
    ParMETIS_ABI_COMPATIBLE)

if(ParMETIS_FOUND)
  if(NOT TARGET METIS::METIS)
    add_library(METIS::METIS UNKNOWN IMPORTED)
    set_target_properties(METIS::METIS PROPERTIES
      IMPORTED_LOCATION "${METIS_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIR}")
  endif()

  if(NOT TARGET ParMETIS::ParMETIS)
    add_library(ParMETIS::ParMETIS UNKNOWN IMPORTED)
    set_target_properties(ParMETIS::ParMETIS PROPERTIES
      IMPORTED_LOCATION "${ParMETIS_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ParMETIS_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "METIS::METIS")
  endif()
endif()

mark_as_advanced(
  ParMETIS_INCLUDE_DIR
  METIS_INCLUDE_DIR
  ParMETIS_LIBRARY
  METIS_LIBRARY)
