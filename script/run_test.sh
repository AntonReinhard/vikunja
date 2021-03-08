#!/bin/bash

echo "container version -> CONTAINER_VERSION"

# the default build type is Release
# if neccesary, you can rerun the pipeline with another build type-> https://docs.gitlab.com/ee/ci/pipelines.html#manually-executing-pipelines
# to change the build type, you must set the environment variable VIKUNJA_BUILD_TYPE

if [[ ! -v VIKUNJA_BUILD_TYPE ]] ; then
    VIKUNJA_BUILD_TYPE=Release ;
fi

###################################################
# cmake config builder
###################################################

VIKUNJA_CONST_ARGS=""
VIKUNJA_CONST_ARGS="${VIKUNJA_CONST_ARGS} -DCMAKE_BUILD_TYPE=${VIKUNJA_BUILD_TYPE}"
VIKUNJA_CONST_ARGS="${VIKUNJA_CONST_ARGS} ${VIKUNJA_CMAKE_ARGS}"

CMAKE_CONFIGS=()
for CXX_VERSION in $VIKUNJA_CXX; do
    for BOOST_VERSION in ${VIKUNJA_BOOST_VERSIONS}; do
	for ACC in ${ALPAKA_ACCS}; do
	    CMAKE_CONFIGS+=("${VIKUNJA_CONST_ARGS} -DCMAKE_CXX_COMPILER=${CXX_VERSION} -DBOOST_ROOT=/opt/boost/${BOOST_VERSION} -D${ACC}=ON")
	done
    done
done

###################################################
# build an run tests
###################################################

# use one build directory for all build configurations
mkdir build
cd build

# ALPAKA_ACCS contains the backends, which are used for each build
# the backends are set in the sepcialized base jobs .base_gcc,.base_clang and.base_cuda
for CONFIG in $(seq 0 $((${#CMAKE_CONFIGS[*]} - 1))); do
    CMAKE_ARGS=${CMAKE_CONFIGS[$CONFIG]}
    echo -e "\033[0;32m///////////////////////////////////////////////////"
    echo "number of processor threads -> $(nproc)"
    cmake --version | head -n 1
    echo "CMAKE_ARGS -> ${CMAKE_ARGS}"
    echo -e "/////////////////////////////////////////////////// \033[0m \n\n"
done
