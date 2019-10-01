//https://leonardoaraujosantos.gitbooks.io/opencl/content/bigger_matrix_multiply_problem.html
//http://gpgpu-computing4.blogspot.com/2009/09/matrix-multiplication-1.html

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

// #define WA 3
// #define HA 3
// #define WB 3
// #define HB 3
// #define WC 3
// #define HC 3

#define HA 5
#define WA 1000000

#define HB 1000000
#define WB 5

#define HC 5
#define WC 5


// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

std::string load_from_file(const std::string &path)
{

    auto close_file = [](FILE *f) { fclose(f); };

    auto holder = std::unique_ptr<FILE, decltype(close_file)>(fopen(path.c_str(), "rb"), close_file);
    if (!holder)
        return "";

    FILE *f = holder.get();

    // in C++17 following lines can be folded into std::filesystem::file_size invocation
    if (fseek(f, 0, SEEK_END) < 0)
        return "";

    const long size = ftell(f);
    if (size < 0)
        return "";

    if (fseek(f, 0, SEEK_SET) < 0)
        return "";

    std::string res;
    res.resize(size);

    // C++17 defines .data() which returns a non-const pointer
    fread(const_cast<char *>(res.data()), 1, size, f);

    return res;
}

int main()
{

    /////////////////////////////////////// PLATFORM and DEVICE Setup ////////////////////////////////////////////////////

    /// 1 - get all platforms (drivers) and set DEFAULT_PLATFORM
    std::vector<cl::Platform> ALL_PLATFORMS;
    cl::Platform::get(&ALL_PLATFORMS);
    if (ALL_PLATFORMS.size() == 0)
    {
        std::cout << " Error finding the platform. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform DEFAULT_PLATFORM = ALL_PLATFORMS[0];
    std::cout << "Using platform: " << DEFAULT_PLATFORM.getInfo<CL_PLATFORM_NAME>() << "\n";

    /// 2 - get the available devices and set DEFAULT_DEVICE
    std::vector<cl::Device> ALL_DEVICES;
    DEFAULT_PLATFORM.getDevices(CL_DEVICE_TYPE_ALL, &ALL_DEVICES);
    if (ALL_DEVICES.size() == 0)
    {
        std::cout << " Error finding the devices. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device DEFAULT_DEVICE = ALL_DEVICES[1]; // use device[1] because that's a GPU; device[0] is the CPU
    std::cout << "Using device: " << DEFAULT_DEVICE.getInfo<CL_DEVICE_NAME>() << "\n";

    // 3 - set the "context" according the the DEFAULT_DEVICE and DEFAULT_PLATFORM
    cl::Context context({DEFAULT_DEVICE});

    /////////////////////////////////////// Loading and Building Kernels ////////////////////////////////////////////////////

    //Load kernels from file
    std::string kernel_code_MatrixMul = load_from_file("kernels/matrixMul.cl");

    // This "sources" variable contains the list of "kernel source codes"
    cl::Program::Sources sources;
    sources.push_back({kernel_code_MatrixMul.c_str(), kernel_code_MatrixMul.length()});

    //Next, building the "kernel program"
    cl::Program program(context, sources);
    if (program.build({DEFAULT_DEVICE}) != CL_SUCCESS)
    {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(DEFAULT_DEVICE) << "\n";
        exit(1);
    }

    /////////////////////////////////////// Defining Data Variables ////////////////////////////////////////////////////

    // set seed for rand()
    // srand(2006);

    // 1. allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);

    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // 2. initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // 3. print out A and B
    /*  
    printf("\n\nMatrix A\n");
    for (int i = 0; i < size_A; i++)
    {
        printf("%f ", h_A[i]);
        if (((i + 1) % WA) == 0)
            printf("\n");
    }

    printf("\n\nMatrix B\n");
    for (int i = 0; i < size_B; i++)
    {
        printf("%f ", h_B[i]);
        if (((i + 1) % WB) == 0)
            printf("\n");
    } 
    */

    // 4. allocate host memory for the result C
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float *h_C = (float *)malloc(mem_size_C);

    /////////////////////////////////////// Creating Buffers and adding to CommandQueue ////////////////////////////////////////////////////

    cl::Buffer d_A(context, CL_MEM_READ_WRITE, mem_size_A);
    cl::Buffer d_B(context, CL_MEM_READ_WRITE, mem_size_B);
    cl::Buffer d_C(context, CL_MEM_READ_WRITE, mem_size_C);

    //CommandQueue
    cl::CommandQueue queue(context, DEFAULT_DEVICE); //create queue to which we will push commands for the device.

    //Adding to CommandQueue
    queue.enqueueWriteBuffer(d_A, CL_TRUE, 0, mem_size_A, h_A); //copy data from arrays A and B to buffer_A and buffer_B which represent memory on the device:
    queue.enqueueWriteBuffer(d_B, CL_TRUE, 0, mem_size_B, h_B);

    /////////////////////////////////////// Running Kernels in CommandQueue ////////////////////////////////////////////////////
    /*
        * run the kernel. We do this with KernelFunctor which runs the kernel on the device.
        * Take a look at the "simple_add" this is the name of our kernel we wrote before.
        * You can see the number 10. This corresponds to number of threads we want to run (our array size is 10):
    */

    //Set the arguments of each kernel and add them into CommandQueue
    cl::Kernel kernel_add = cl::Kernel(program, "matrixMul");

    kernel_add.setArg(0, d_C);
    kernel_add.setArg(1, d_A);
    kernel_add.setArg(2, d_B);
    kernel_add.setArg(3, WA);
    kernel_add.setArg(4, WB);

    // queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(10), cl::NullRange);
    queue.enqueueNDRangeKernel(kernel_add,        //kernel
                               cl::NullRange,     // NDRange &offset
                               cl::NDRange(HC, WC), //NDRange &global     //globalWorkSize[0] = 3; globalWorkSize[1] = 3;
                               cl::NullRange,
                               //cl::NDRange(HC, WC), //NDRange &local      //localWorkSize[0] = 3; localWorkSize[1] = 3;
                               NULL,              //vector< Event >
                               NULL               //Event *event
    );
    //  (const Kernel &kernel, const NDRange &offset, const NDRange &global, const NDRange &local=NullRange, const vector< Event > *events=NULL, Event *event=NULL)
    queue.finish();

    /////////////////////////////////////// Retrieve Output Buffer ////////////////////////////////////////////////////

    queue.enqueueReadBuffer(d_C,        //buffer
                            CL_TRUE,    //blocking
                            0,          //offset
                            mem_size_C, //size
                            h_C,        //ptr
                            NULL, NULL);
    // (const Buffer &buffer, cl_bool blocking, size_type offset, size_type size, void *ptr, const vector< Event > *events=NULL, Event *event=NULL) const

    /////////////////////////////////////// Display Output ////////////////////////////////////////////////////

    // 9. print out the results
    printf("\n\nMatrix C (Results)\n");
    for (int i = 0; i < size_C; i++)
    {
        printf("%f ", h_C[i]);
        if (((i + 1) % WC) == 0)
            printf("\n");
    }
    printf("\n");

    // 10. clean up memory
    // free(h_A);
    // free(h_B);
    // free(h_C);

    // clReleaseMemObject(d_A);
    // clReleaseMemObject(d_C);
    // clReleaseMemObject(d_B);

    // free(clDevices);
    // free(clMatrixMul);
    // clReleaseContext(clGPUContext);
    // clReleaseKernel(clKernel);
    // clReleaseProgram(clProgram);
    // clReleaseCommandQueue(clCommandQue);

    return 0;
}
