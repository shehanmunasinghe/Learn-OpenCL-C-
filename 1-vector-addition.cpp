#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

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
    std::vector<cl::Platform> ALL_PLATFORMS; cl::Platform::get(&ALL_PLATFORMS);
    if (ALL_PLATFORMS.size() == 0){
        std::cout << " Error finding the platform. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform DEFAULT_PLATFORM = ALL_PLATFORMS[0];
    std::cout << "Using platform: " << DEFAULT_PLATFORM.getInfo<CL_PLATFORM_NAME>() << "\n";

    /// 2 - get the available devices and set DEFAULT_DEVICE
    std::vector<cl::Device> ALL_DEVICES; DEFAULT_PLATFORM.getDevices(CL_DEVICE_TYPE_ALL, &ALL_DEVICES);
    if (ALL_DEVICES.size() == 0){
        std::cout << " Error finding the devices. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device DEFAULT_DEVICE = ALL_DEVICES[1]; // use device[1] because that's a GPU; device[0] is the CPU
    std::cout << "Using device: " << DEFAULT_DEVICE.getInfo<CL_DEVICE_NAME>() << "\n";

    // 3 - set the "context" according the the DEFAULT_DEVICE and DEFAULT_PLATFORM
    cl::Context context({DEFAULT_DEVICE});





    /////////////////////////////////////// Loading and Building Kernels ////////////////////////////////////////////////////

    //Load kernels from file
    std::string kernel_code = load_from_file("kernels/vector_add_kernel.cl");  
    
    // This "sources" variable contains the list of "kernel source codes"
    cl::Program::Sources sources;
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    //Next, building the "kernel program"
    cl::Program program(context, sources);
    if (program.build({DEFAULT_DEVICE}) != CL_SUCCESS)
    {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(DEFAULT_DEVICE) << "\n";
        exit(1);
    }






    /////////////////////////////////////// Defining Data Variables ////////////////////////////////////////////////////
    int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

    /////////////////////////////////////// Creating Buffers and adding to CommandQueue ////////////////////////////////////////////////////
    
    //Defining Buffers
    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * 10);


    //CommandQueue
    cl::CommandQueue queue(context, DEFAULT_DEVICE); //create queue to which we will push commands for the device.
    
    //Adding to CommandQueue
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * 10, A); //copy data from arrays A and B to buffer_A and buffer_B which represent memory on the device:
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * 10, B);




    /////////////////////////////////////// Running Kernels in CommandQueue ////////////////////////////////////////////////////
    /*
        * run the kernel. We do this with KernelFunctor which runs the kernel on the device.
        * Take a look at the "simple_add" this is the name of our kernel we wrote before.
        * You can see the number 10. This corresponds to number of threads we want to run (our array size is 10):
    */

    //Set the arguments of each kernel and add them into CommandQueue
    cl::Kernel kernel_add = cl::Kernel(program, "simple_add");
    kernel_add.setArg(0, buffer_A);
    kernel_add.setArg(1, buffer_B);
    kernel_add.setArg(2, buffer_C);
    queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(10), cl::NullRange);
    queue.finish();



    /////////////////////////////////////// Retrieve Output Buffer ////////////////////////////////////////////////////

    int C[10];
    //read result C from the device to array C	//transfer data from the device to our program (host)
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * 10, C);

    /////////////////////////////////////// Display Output ////////////////////////////////////////////////////
    std::cout << " result: \n";
    for (int i = 0; i < 10; i++)
    {
        std::cout << C[i] << " ";
    }

    return 0;
}
