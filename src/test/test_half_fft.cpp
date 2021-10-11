#include <cuda_runtime.h>
#include <other_function_library.hpp>
#include <kernal_wrapper.hpp>
#include <cufft_wrapper.hpp>

int main(int argc, char *argv[]) {
    short *input_half;
    cudaMallocManaged((void**)&input_half,sizeof(unsigned char)*100);
    float input_float[34]={1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.,32.,33.,34.};
   
    /*float_2_half((void*)input_float,(void*)input_half,0,34);
    print_data_half_for_copy(input_half,0,34);
    fft_1d_half((void*)input_half, (void*)input_half,4,1);
    print_data_half_for_copy(input_half,0,34);
    
    float_2_half((void*)input_float,(void*)input_half,0,34);
    print_data_half_for_copy(input_half,0,34);
    fft_1d_half((void*)input_half, (void*)input_half,8,1);
    print_data_half_for_copy(input_half,0,34);
    
    float_2_half((void*)input_float,(void*)input_half,0,34);
    print_data_half_for_copy(input_half,0,34);
    fft_1d_half((void*)input_half, (void*)input_half,16,1);
    print_data_half_for_copy(input_half,0,34);
    
    float_2_half((void*)input_float,(void*)input_half,0,34);
    print_data_half_for_copy(input_half,0,34);
    fft_1d_half((void*)input_half, (void*)input_half,32,1);
    print_data_half_for_copy(input_half,0,34);*/
    print_data_float(input_float,0,34);
    
}
