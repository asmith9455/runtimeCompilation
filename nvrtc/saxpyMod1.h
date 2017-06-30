#pragma once
#include <nvrtc.h>
#include <cuda.h>
#include <iostream>
#include <string>


//example from the user guide: http://docs.nvidia.com/cuda/nvrtc/index.html#example-saxpy
//modified to take a string arg from the caller
//visual studio project settings set for CUDA 8.0 libs
//sample arg: "a * x[tid] + y[tid]"
void runSaxpyMod1(std::string toEval);