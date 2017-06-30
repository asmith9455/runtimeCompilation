//example from the user guide: http://docs.nvidia.com/cuda/nvrtc/index.html#example-saxpy
//visual studio project settings set for CUDA 8.0 libs
#include <saxpyMod2.h>
#include <Node.h>



#define NUM_THREADS 1
#define NUM_BLOCKS 1



#define NVRTC_SAFE_CALL(x) \
 do { \
 nvrtcResult result = x; \
 if (result != NVRTC_SUCCESS) { \
 std::cerr << "\nerror: " #x " failed with error " \
 << nvrtcGetErrorString(result) << '\n'; \
 exit(1); \
 } \
 } while(0)




#define CUDA_SAFE_CALL(x) \
 do { \
 CUresult result = x; \
 if (result != CUDA_SUCCESS) { \
 const char *msg; \
 cuGetErrorName(result, &msg); \
 std::cerr << "\nerror: " #x " failed with error " \
 << msg << '\n'; \
 exit(1); \
 } \
 } while(0)




void runSaxpyMod2(std::string toEval)
{

	std::string kernelUDF = "struct Node{int x, y, u;int r; Node* westNode;}; \n\
						extern \"C\" __global__ \n\
						void saxpy(Node* dnodes, size_t n) \n\
						{ \n\
						 size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \n\
						 Node node = dnodes[tid];\
						 if (tid < n) { \n\
						 dnodes[tid].r = ";


	std::string kernelSetPtrs = "struct Node{int x, y, u;int r; Node* westNode;}; \n\
						extern \"C\" __global__ \n\
						void saxpy(Node* dnodes, size_t n) \n\
						{ \n\
						 size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \n\
						 if (tid < n) { \n\
							if (tid == 0)\
								dnodes[tid].westNode = &(dnodes[n-1]);\
							else\
								dnodes[tid].westNode = &(dnodes[tid-1]);\
						}\
						}";


	std::string tmp2 = " } }";

	std::string argFinal1 = kernelUDF + toEval + tmp2;

	// Create an instance of nvrtcProgram with the SAXPY code string.
	nvrtcProgram prog;
	NVRTC_SAFE_CALL(
		nvrtcCreateProgram(&prog, // prog
			argFinal1.c_str(), // buffer
			"saxpy.cu", // name
			0, // numHeaders
			NULL, // headers
			NULL)); // includeNames

	nvrtcProgram prog2;
	NVRTC_SAFE_CALL(
		nvrtcCreateProgram(&prog2, // prog
			kernelSetPtrs.c_str(), // buffer
			"saxpySetPtrs.cu", // name
			0, // numHeaders
			NULL, // headers
			NULL)); // includeNames
					// Compile the program for compute_20 with fmad disabled.



	const char *opts[] = { "--gpu-architecture=compute_20",
		"--fmad=false", "-default-device" };

	//NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, "Node"));
	nvrtcResult compileResult = nvrtcCompileProgram(prog, // prog
		2, // numOptions
		opts); // options

	nvrtcResult compileResult2 = nvrtcCompileProgram(prog2, // prog
		2, // numOptions
		opts); // options
			   // Obtain compilation log from the program.
	size_t logSize, logSize2;
	NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
	NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog2, &logSize2));
	char *log = new char[logSize];
	char *log2 = new char[logSize2];
	NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
	NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog2, log2));
	std::cout << "compilelog1: " << log << '\n';
	std::cout << "compilelog2: " << log2 << '\n';
	delete[] log;
	delete[] log2;
	if (compileResult != NVRTC_SUCCESS) {
		exit(1);
	}
	// Obtain PTX from the program.
	size_t ptxSize;
	NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
	char *ptx = new char[ptxSize];
	NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
	// Destroy the program.
	NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
	// Load the generated PTX and get a handle to the SAXPY kernel.
	CUdevice cuDevice;
	CUcontext context;
	CUmodule module;
	CUfunction kernel;
	CUDA_SAFE_CALL(cuInit(0));
	CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
	CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
	CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
	CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "saxpy"));
	// Generate input for execution, and create output buffers.
	size_t n = NUM_THREADS * NUM_BLOCKS;
	size_t bufferSize = n * sizeof(Node);
	//float a = 5.1f;
	//float *hX = new float[n], *hY = new float[n], *hOut = new float[n];
	Node* nodes = new Node[n];
	for (size_t i = 0; i < n; ++i) {
		nodes[i] = Node(1,4,i*5);
		if (i == 0)
			nodes[i].westNode = &(nodes[n-1]);//first element points to last
		else
			nodes[i].westNode = &(nodes[i - 1]);
	}

	CUdeviceptr dnodes;
	CUDA_SAFE_CALL(cuMemAlloc(&dnodes, bufferSize));
	CUDA_SAFE_CALL(cuMemcpyHtoD(dnodes, nodes, bufferSize));
	// Execute SAXPY.
	void *args[] = { &dnodes, &n };
	CUDA_SAFE_CALL(
		cuLaunchKernel(kernel,
			NUM_THREADS, 1, 1, // grid dim
			NUM_BLOCKS, 1, 1, // block dim
			0, NULL, // shared mem and stream
			args, 0)); // arguments
	CUDA_SAFE_CALL(cuCtxSynchronize());
	// Retrieve and print output.
	CUDA_SAFE_CALL(cuMemcpyDtoH(nodes, dnodes, bufferSize));
	std::cout << "expression: " << toEval << std::endl;
	for (size_t i = 0; i < n; ++i) {
		std::cout << "nodes[i].r: " << nodes[i].r << std::endl;
	}
	// Release resources.
	CUDA_SAFE_CALL(cuMemFree(dnodes));
	CUDA_SAFE_CALL(cuModuleUnload(module));
	CUDA_SAFE_CALL(cuCtxDestroy(context));
	delete[] nodes;
}
