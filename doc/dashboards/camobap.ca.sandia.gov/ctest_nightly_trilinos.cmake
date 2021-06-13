
#cmake_minimum_required (VERSION 2.8)
set (CTEST_DO_SUBMIT ON)
set (CTEST_TEST_TYPE Nightly)
set (CTEST_BUILD_OPTION "$ENV{BUILD_OPTION}")

if (1)
  # What to build and test
  IF(CTEST_BUILD_OPTION MATCHES "download")
    set(DOWNLOAD_TRILINOS TRUE)
    message("Downloading Trilinos!") 
  ELSEIF(CTEST_BUILD_OPTION MATCHES "build") 
    set (BUILD_TRILINOS TRUE)
    message("Building Trilinos!") 
  ENDIF()
ENDIF()

# What to build and test
#set (DOWNLOAD_TRILINOS TRUE)
#set (BUILD_TRILINOS TRUE)

# Begin User inputs:
set (CTEST_SITE "camobap.ca.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_BUILD_NAME "fedora34-gcc11.1.1-Trilinos-extended-sts")
set (CTEST_BINARY_NAME build)

set (CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
set (CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")

if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_SOURCE_DIRECTORY}")
endif ()
if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}")
endif ()

configure_file (${CTEST_SCRIPT_DIRECTORY}/CTestConfig.cmake
  ${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake COPYONLY)

set (CTEST_NIGHTLY_START_TIME "01:00:00 UTC")
set (CTEST_CMAKE_COMMAND "cmake")
set (CTEST_COMMAND "ctest -D ${CTEST_TEST_TYPE}")
set (CTEST_FLAGS "-j32")
SET (CTEST_BUILD_FLAGS "-j32")

set (CTEST_DROP_METHOD "https")

if (CTEST_DROP_METHOD STREQUAL "https")
  set (CTEST_DROP_SITE "sems-cdash-son.sandia.gov")
  set (CTEST_PROJECT_NAME "Albany")
  set (CTEST_DROP_LOCATION "/cdash/submit.php?project=Albany")
  set (CTEST_TRIGGER_SITE "")
  set (CTEST_DROP_SITE_CDASH TRUE)
endif ()

find_program (CTEST_GIT_COMMAND NAMES git)

set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)

if (CLEAN_BUILD)
  # Initial cache info
  set (CACHE_CONTENTS "
  SITE:STRING=${CTEST_SITE}
  CMAKE_TYPE:STRING=Release
  CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
  TESTING:BOOL=OFF
  PRODUCT_REPO:STRING=${Albany_REPOSITORY_LOCATION}
  " )

  ctest_empty_binary_directory( "${CTEST_BINARY_DIRECTORY}" )
  file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "${CACHE_CONTENTS}")
endif ()

if (DOWNLOAD_TRILINOS)

  set (CTEST_CHECKOUT_COMMAND)
 
  #
  # Get Trilinos
  #
  
  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Trilinos")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${Trilinos_REPOSITORY_LOCATION} -b develop ${CTEST_SOURCE_DIRECTORY}/Trilinos
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot clone Trilinos repository!")
    endif ()
  endif ()

endif()


ctest_start(${CTEST_TEST_TYPE})

#
# Send the project structure to CDash
#

#if (CTEST_DO_SUBMIT)
#  ctest_submit (FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
#    RETURN_VALUE  HAD_ERROR
#    )

#  if (HAD_ERROR)
#    message(FATAL_ERROR "Cannot submit Albany Project.xml!")
#  endif ()
#endif ()

# 
# Set the common Trilinos config options & build Trilinos
# 

if (BUILD_TRILINOS) 
  message ("ctest state: BUILD_TRILINOS")
  #
  # Configure the Trilinos/SCOREC build
  #

  
  set(TRILINOS_INSTALL ${CTEST_BINARY_DIRECTORY}/TrilinosInstall) 

  set (CONFIGURE_OPTIONS
     "-DCMAKE_INSTALL_PREFIX:PATH=${TRILINOS_INSTALL}"
     "-DCMAKE_BUILD_TYPE:STRING=RELEASE"
     "-DTPL_BLAS_LIBRARIES:FILEPATH=/usr/lib64/libblas.so.3"
     "-DTPL_LAPACK_LIBRARIES:FILEPATH=/usr/lib64/liblapack.so.3"
     "-DTPL_ENABLE_MPI:BOOL=ON"
     "-DMPI_BASE_DIR:PATH=/nightlyCDash/albany-tpls-gcc-11.1.1-openmpi-4.0.5"
     "-DCMAKE_C_COMPILER=/nightlyCDash/albany-tpls-gcc-11.1.1-openmpi-4.0.5/bin/mpicc"
     "-DCMAKE_CXX_COMPILER=/nightlyCDash/albany-tpls-gcc-11.1.1-openmpi-4.0.5/bin/mpicxx"
     "-DCMAKE_Fortran_COMPILER=/nightlyCDash/albany-tpls-gcc-11.1.1-openmpi-4.0.5/bin/mpif90"
     "-DCMAKE_CXX_FLAGS:STRING='-std=gnu++11 -fext-numeric-literals'"
     "-DTPL_ENABLE_Netcdf:BOOL=ON"
     "-DTPL_Netcdf_INCLUDE_DIRS:PATH=/nightlyCDash/albany-tpls-gcc-11.1.1-openmpi-4.0.5/include"
     "-DNetcdf_LIBRARY_DIRS:PATH=/nightlyCDash/albany-tpls-gcc-11.1.1-openmpi-4.0.5/lib"
     "-DTPL_ENABLE_HDF5:BOOL=OFF"
     "-DAmesos2_ENABLE_KLU2:BOOL=ON"
     "-DTPL_ENABLE_Boost:BOOL=ON"
     "-DTPL_ENABLE_BoostLib:BOOL=ON"
     "-DBoost_INCLUDE_DIRS:FILEPATH=/usr/include"
     "-DBoost_LIBRARY_DIRS:FILEPATH=/usr/lib64"
     "-DBoostLib_INCLUDE_DIRS:FILEPATH=/usr/include"
     "-DBoostLib_LIBRARY_DIRS:FILEPATH=/usr/lib64"
     "-DTrilinos_ENABLE_TESTS:BOOL=OFF"
     "-DPiro_ENABLE_TESTS:BOOL=OFF"
     "-DTeuchos_ENABLE_TESTS:BOOL=ON"
     "-DTeuchos_ENABLE_EXAMPLES:BOOL=ON"
     "-DKokkos_ENABLE_TESTS:BOOL=OFF"
     "-DKokkos_ENABLE_EXAMPLES:BOOL=OFF"
     "-DRythmos_ENABLE_TESTS:BOOL=OFF"
     "-DROL_ENABLE_TESTS:BOOL=OFF"
     "-DTrilinos_ENABLE_EXAMPLES:BOOL=OFF"
     "-DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF"
     "-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF"
     "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
     "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=OFF"
     "-DTrilinos_ENABLE_Amesos:BOOL=ON"
     "-DTrilinos_ENABLE_Amesos2:BOOL=ON"
     "-DTrilinos_ENABLE_Anasazi:BOOL=ON"
     "-DTrilinos_ENABLE_AztecOO:BOOL=ON"
     "-DTrilinos_ENABLE_Belos:BOOL=ON"
     "-DBelos_ENABLE_TESTS:BOOL=OFF"
     "-DBelos_ENABLE_EXAMPLES:BOOL=OFF"
     "-DTrilinos_ENABLE_COMPLEX_DOUBLE:BOOL=ON"
     "-DTrilinos_ENABLE_Epetra:BOOL=ON"
     "-DTrilinos_ENABLE_EpetraExt:BOOL=ON"
     "-DTrilinos_ENABLE_FEI:BOOL=OFF"
     "-DTrilinos_ENABLE_Ifpack:BOOL=ON"
     "-DTrilinos_ENABLE_Ifpack2:BOOL=ON"
     "-DIfpack2_ENABLE_TESTS:BOOL=OFF"
     "-DIfpack2_ENABLE_EXAMPLES:BOOL=OFF"
     "-DTrilinos_ENABLE_Intrepid:BOOL=ON"
     "-DTrilinos_ENABLE_Intrepid2:BOOL=ON"
     "-DTrilinos_ENABLE_Kokkos:BOOL=ON"
     "-DKokkos_ENABLE_SERIAL:BOOL=ON"
     "-DKokkos_ENABLE_OPENMP:BOOL=OFF"
     "-DKokkos_ENABLE_PTHREAD:BOOL=OFF"
     "-DTrilinos_ENABLE_OpenMP:BOOL=OFF"
     "-DTrilinos_ENABLE_MiniTensor:BOOL=ON"
     "-DTrilinos_ENABLE_ML:BOOL=ON"
     "-DTrilinos_ENABLE_MueLu:BOOL=OFF"
     "-DTrilinos_ENABLE_NOX:BOOL=ON"
     "-DTrilinos_ENABLE_Pamgen:BOOL=ON"
     "-DTrilinos_ENABLE_PanzerExprEval:BOOL=ON"
     "-DTrilinos_ENABLE_Phalanx:BOOL=ON"
     "-DTrilinos_ENABLE_Piro:BOOL=ON"
     "-DPiro_ENABLE_TESTS:BOOL=OFF"
     "-DPiro_ENABLE_EXAMPLES:BOOL=OFF"
     "-DAnasazi_ENABLE_RBGen:BOOL=OFF"
     "-DTrilinos_ENABLE_ROL:BOOL=ON"
     "-DTrilinos_ENABLE_Rythmos:BOOL=ON"
     "-DTrilinos_ENABLE_Sacado:BOOL=ON"
     "-DTrilinos_ENABLE_SEACAS:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASAprepro_lib:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASConjoin:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASEjoin:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASEpu:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASExodiff:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASExodus:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASIoss:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASNemslice:BOOL=ON"
     "-DTrilinos_ENABLE_Shards:BOOL=ON"
     "-DTrilinos_ENABLE_ShyLU_DDFROSch:BOOL=ON"
     "-DTrilinos_ENABLE_STKUnit_tests:BOOL=ON"
     "-DTrilinos_ENABLE_STKIO:BOOL=ON"
     "-DTrilinos_ENABLE_STKMesh:BOOL=ON"
     "-DTrilinos_ENABLE_STKExprEval:BOOL=ON"
     "-DTrilinos_ENABLE_Stokhos:BOOL=OFF"
     "-DTrilinos_ENABLE_Stratimikos:BOOL=ON"
     "-DTrilinos_ENABLE_Teko:BOOL=ON"
     "-DTrilinos_ENABLE_Teuchos:BOOL=ON"
     "-DTrilinos_ENABLE_Thyra:BOOL=ON"
     "-DTrilinos_ENABLE_ThyraTpetraAdapters:BOOL=ON"
     "-DTrilinos_ENABLE_ThyraEpetraAdapters:BOOL=ON"
     "-DTrilinos_ENABLE_Tpetra:BOOL=ON"
     "-DTpetra_ENABLE_TESTS:BOOL=OFF"
     "-DTpetra_ENABLE_EXAMPLES:BOOL=OFF"
     "-DTrilinos_ENABLE_TrilinosCouplings:BOOL=ON"
     "-DTrilinos_ENABLE_TriKota:BOOL=OFF"
     "-DTrilinos_ENABLE_Zoltan:BOOL=ON"
     "-DTrilinos_ENABLE_Zoltan2:BOOL=ON"
     "-DZoltan_ENABLE_ULONG_IDS:BOOL=OFF"
     "-DZOLTAN_BUILD_ZFDRIVE:BOOL=OFF"
     "-DTrilinos_ENABLE_DEBUG:BOOL=OFF"
     "-DPhalanx_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
     "-DStratimikos_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
     "-DTempus_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
     "-DPhalanx_KOKKOS_DEVICE_TYPE:STRING='SERIAL'"
     "-DPhalanx_INDEX_SIZE_TYPE:STRING='INT'"
     "-DPhalanx_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
     "-DTrilinos_ENABLE_SCOREC:BOOL=OFF"
     "-DTrilinos_ENABLE_Tempus:BOOL=ON"
     "-DTempus_ENABLE_TESTS:BOOL=OFF"
     "-DTempus_ENABLE_EXAMPLES:BOOL=OFF"
     "-DTempus_ENABLE_EXPLICIT_INSTANTIATION:BOOL=OFF"
     "-DTPL_ENABLE_MOAB:BOOL=OFF"
     "-DTPL_ENABLE_Libmesh:BOOL=OFF"
     "-DTPL_Netcdf_PARALLEL:BOOL=ON"
     "-DTPL_ENABLE_Matio=OFF"
     "-DTPL_ENABLE_X11=OFF"
     "-DTrilinos_ENABLE_CXX11:BOOL=ON"
     "-DTPL_FIND_SHARED_LIBS:BOOL=ON"
     "-DBUILD_SHARED_LIBS:BOOL=ON"
     "-DTrilinos_LINK_SEARCH_START_STATIC:BOOL=OFF"
     "-DMPI_EXEC=mpirun"
     "-DPhalanx_ALLOW_MULTIPLE_EVALUATORS_FOR_SAME_FIELD:BOOL=ON"
     "-DKOKKOS_ENABLE_LIBDL:BOOL=ON"
     "-DTpetra_ENABLE_DEPRECATED_CODE=ON"
     "-DXpetra_ENABLE_DEPRECATED_CODE=ON"
     "-DTrilinos_ENABLE_LONG_DOUBLE:BOOL=ON"
     "-DTPL_ENABLE_quadmath:BOOL=ON"
     "-DTPL_quadmath_LIBRARIES=/usr/lib64/libquadmath.so.0"
     "-Dquadmath_INCLUDE_DIRS:FILEPATH=/usr/lib/gcc/x86_64-redhat-linux/10/include"
     "-DTpetra_ENABLE_quadmath:BOOL=ON"
     "-DTpetra_INST_FLOAT128:BOOL=OFF"
  )

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuild")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/TriBuild)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
    SOURCE "/nightlyCDash/repos/Trilinos"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Trilinos configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure Trilinos build!")
  endif ()

  #
  # Build the rest of Trilinos and install everything
  #

  #set (CTEST_BUILD_TARGET all)
  set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Trilinos build results!")
    endif ()

  endif ()

  if (HAD_ERROR)
    message ("Cannot build Trilinos!")
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message ("Encountered build errors in Trilinos build. Exiting!")
  endif ()
  
  #
  # Run Trilinos tests
  #

  set (CTEST_TEST_TIMEOUT 1500)
  CTEST_TEST (
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
    RETURN_VALUE HAD_ERROR)

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test RETURN_VALUE S_HAD_ERROR)

    if (S_HAD_ERROR)
      message ("Cannot submit Trilinos test results!")
    endif ()
  endif ()

endif()
