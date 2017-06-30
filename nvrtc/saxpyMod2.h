#pragma once
#include <nvrtc.h>
#include <cuda.h>
#include <iostream>
#include <string>


//example from the user guide: http://docs.nvidia.com/cuda/nvrtc/index.html#example-saxpy
//visual studio project settings set for CUDA 8.0 libs

void runSaxpyMod2(std::string toEval);