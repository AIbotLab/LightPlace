#!/bin/bash
help_f(){
    echo "!!! you should include an external parameter"
    echo "!!! you can choose \"rebuild\" or \"make\" only"
    echo "    like:   $0 rebuild  or  $0 make "
}

make_f(){
    cd "build" 
    make -j8
    make install
}
rebuild_f(){
    rm -rf "build"
    mkdir "build"
    cd "build" 
    cmake .. -DCMAKE_INSTALL_PREFIX=/home/cjq/work/ISPDPlace/eDensity3d/build -DPython_EXECUTABLE=$(which python)
    make -j8
    make install
}

if [ $# != 1 ]
then
    help_f
    exit
fi

option=$1

case $option in
    "make") {
        make_f
    };;
    "rebuild") {
        rebuild_f
    };;
    *) {
        help_f
        exit
    };;
esac
