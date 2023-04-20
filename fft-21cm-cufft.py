#!/usr/bin/env python
"""Example program showing how to read from a ringbuffer using an iterator."""
import numpy as np
import cupy as cp
import skcuda.cufft as cufft
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as gpudrv
import time
from psrdada.reader import Reader, PSRDadaError
from pycuda.cumath import sqrt, log10
import gc
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule

# # unsigned integer
# utofloat = ElementwiseKernel(
#         "unsigned char *x, float *y",
#         "y[i] = (float) x[i] + 255.0",
#         "utofloat",
#         )

# signed integer
utofloat = ElementwiseKernel(
        "signed char *x, float *y",
        "y[i] = (float) x[i]",
        "utofloat",
        )
# kernelmodel = '''
#         #include <stdio.h>
#         #include <pycuda-complex.hpp>
#         #include <cuComplex.h>
#     #define Bx blockIdx.x
#     #define By blockIdx.y
#     #define Bz blockIdx.z
#     #define Tx threadIdx.x
#     #define Ty threadIdx.y
#     #define Tz threadIdx.z
#         __global__ void power(pycuda::complex<float> *x, float *power, int nf, int batchsize)
#         {
#         int offset = Tx + blockDim.x * Ty + Bx*blockDim.x*blockDim.y;
#             for (int j=0; j < batchsize; j++)
#             {
#                 int i = offset + nf * j;
#         power[offset] += abs(x[i] * pycuda::conj(x[i]));
#             }
#         __syncthreads();

#         }
# '''

kernelmodel ='''
#include <stdio.h>
#include <pycuda-complex.hpp>
#include <cuComplex.h>
#define Bx blockIdx.x
#define By blockIdx.y
#define Bz blockIdx.z
#define Tx threadIdx.x
#define Ty threadIdx.y
#define Tz threadIdx.z

__global__ void power(pycuda::complex<float> *x, float *power, int nf, int batchsize)
{
    int offset = Tx + blockDim.x * Ty + Bx*blockDim.x*blockDim.y;
    for (int j=0; j < batchsize; j++)
    {
        int i = offset + nf * j;
        atomicAdd(&power[offset], abs(x[i] * pycuda::conj(x[i])));
    }
}
'''


def power():
    mod = SourceModule(kernelmodel, options=["--expt-relaxed-constexpr"])
    power_fun = mod.get_function("power")
    return power_fun
Power = power()

def read_untill_end():
    try:
        # block and grid
        int_t = 0
        tsamp = 1./1e9
        # tsamp = 1./1e6
        dsize = 67108864*2*2
        dt = dsize*tsamp
        ind = 0
        fftlen = 8192*8
        nf = fftlen//2
        batchsize = dsize // (fftlen)
        block = (32,32,1)
        grid = (nf//1024,1,1)  #nf//32//32
        # 设置随机数种子，以便结果可重复
        # np.random.seed(123)

        # 生成一个形状为(3-batchsize, 4-nf)的随机复数数组，每个复数的实部和虚部都在[-1, 1)之间
        # complex_arr = np.random.uniform(-1, 1, size=(batchsize, nf)) + 1j * np.random.uniform(-1, 1, size=(batchsize, nf))
        # input_data = complex_arr.astype(np.complex64).ravel()

        # # load data
        # input_data = np.load("input_data.npy")
        # input_data = np.asarray(input_data).astype(np.float32)
        # input_gpu = gpuarray.to_gpu(input_data)

        # fft plan
        stm = time.time()
        plan = cufft.cufftPlan1d(fftlen, cufft.CUFFT_R2C, batchsize)
        print("fft plan time:",time.time()-stm) #0.05 sec


        stm = time.time()
        # fft 输出数据 @ gpu
        output_fft = np.zeros(batchsize * (nf+1), dtype=np.complex64)
        output_gpu = gpuarray.to_gpu(output_fft)
        input_float = np.zeros(dsize, dtype=np.float32)
        input_gpu = gpuarray.to_gpu(input_float)


        # print("\ninput_gpu:",input_gpu.get())
        # sumall = np.sum(abs(input_gpu.reshape(batchsize,nf).get()), axis=0)
        # print("\ninput_gpu module sum:", sumall)


        # fft power @gpu
        # specpower_ = np.empty(nf, dtype=np.float32)
        # specpower = gpuarray.to_gpu(specpower_)
        specpower_ = np.zeros(nf+1, dtype=np.float32)
        specpower = gpuarray.to_gpu(specpower_)


        # 求 fft 功率并按照batchsize累加

        # print("type Power in:",type(input_gpu))
        # print("type Power out:",type(specpower))
        # Power(input_gpu, specpower,  np.int32(nf), np.int32(batchsize), grid=grid, block=block)


        # check Power input array size
        # print("output_gpu size",output_gpu.size)
        # print("specpower size",specpower.size)

        # # get data to cpu from gpu
        vf = np.fft.fftfreq(fftlen, d=tsamp)[:nf+1]/1e6   #unit in MHz
        vf = np.fft.fftfreq(fftlen, d=tsamp)[:nf+1]/1e3   #unit in kHz
        Spec = gpuarray.zeros((nf + 1), dtype=np.float32)
        print("initialize time:",time.time()-stm)  # 0.39 sec
        # np.log10(specpower.get())

        # print("\ngpu module kernel result:")
        # print("Spec:",Spec)
        # print("Spec shape",Spec.shape)

        reader = Reader(0xdada)
        while not reader.isEndOfData:
            print("%d-th data block\n"%(ind))
            # stm = time.time()
            page = reader.getNextPage()
            #input_data = np.asarray(page).astype(np.float32)

            # page_gpu = gpuarray.to_gpu(np.asarray(page, dtype='u1'))
            page_gpu = gpuarray.to_gpu(np.asarray(page, dtype='i1'))
            # print("upload data time:",time.time()-stm) # 0.24 sec
            utofloat(page_gpu, input_gpu)

            # stm = time.time()
            cufft.cufftExecR2C(plan, int(input_gpu.gpudata), int(output_gpu.gpudata))
            # print("fft time:",time.time()-stm)
            
            # time 
            # stm = time.time()
            Power(output_gpu, specpower,  np.int32(nf + 1), np.int32(batchsize), grid=grid, block=block)
            # print("Power time:",time.time()-stm) # 0.16 sec

            # stm = time.time()
            # Spec= Spec + specpower   # 0.1 sec
            # np.save("20230316-cufft-spec", output_gpu.get())   ############ remove after
            Spec += specpower
            # print("add time:",time.time()-stm) 0.00012 sec

            # print("Spec shape:",Spec.shape)
            # print("Spec data:",Spec)
            # print("log10 (Spec):", log10(Spec))

            # print integration time
            int_t += dt
            print("Integration time is:%4f sec"%(int_t))
            # print("add time:",time.time()-stm)   # 0.12 sec
            reader.markCleared()
            ind +=1
            # print("loop time:",time.time()-stm)

    except PSRDadaError as e:
        print(f"PSRDada error occurred: {e}")
    finally:
        log_power = (log10(Spec)).get()
        np.savez("Spec20230414-cufft", freqs=vf, power=log_power)
        print("Done!")
        reader.disconnect()


if __name__ == '__main__':    
    read_untill_end()

## 导致内存增长的原因是 GPU数据被下载到CPU了
# specpower.get()
## 导致 pycuda._driver.LogicError:cuFuncSetBlockShape failed: invalid resource handle错误：
# vf = cp.fft.fftfreq(fftlen, d=tsamp)[:nf+1]/1e6   #unit in MHz 
