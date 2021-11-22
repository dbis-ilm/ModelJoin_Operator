#!/bin/bash

dir=$(pwd)

if ! [[ "$BUILDMODE" != "RELEASE" || "$BUILDMODE" != "DEBUG" ]]; then
    echo "BUILDMODE not set. Please set to RELEASE or DEBUG."
    return 0
fi

if [ "$BUILDMODE" = "RELEASE" ]; then
    export BUILD_DIR="$dir/release"
    type="Release"
fi

if [ "$BUILDMODE" = "DEBUG" ]; then
    export BUILD_DIR="$dir/debug"
    type="Debug"
fi

rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake -DCMAKE_BUILD_TYPE=$type -B$BUILD_DIR $dir
make

export TEST_DIR="$dir/test"

cd $dir
