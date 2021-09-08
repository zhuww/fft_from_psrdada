#include <filesystem>
#include <iostream>
#include <fstream>

void readfile(signed char *input_char_cpu,signed char *input_char_gpu,std::string * file_list,long long int size)
{
    static std::ifstream original_data;
    static int file_number=0;
    static unsigned long long int file_remain_size=0;
    unsigned long long int read_remain_size=size;
    unsigned long long int read_size=0;
    //打开文件

    while(read_remain_size>0)
    {
        if(!original_data.is_open())
        {
            original_data.open(file_list[file_number].c_str(),std::ios::in|std::ios::binary);
            if(original_data.is_open()) std::cout << "Succeed opening input file "<<file_list[file_number]<<"\n";
            file_remain_size=std::filesystem::file_size(file_list[file_number]);
            std::cout << "File size is "<<file_remain_size<<"\n";
        }

        if(read_remain_size<file_remain_size)
        {
            original_data.read((char *) input_char_cpu+read_size, read_remain_size* sizeof(char));
            file_remain_size-=read_remain_size;
            read_size+=read_remain_size;
            read_remain_size=0;
        }
        else
        {
            original_data.read((char *) input_char_cpu+read_size, file_remain_size* sizeof(char));
            read_remain_size-=file_remain_size;
            read_size+=file_remain_size;
            file_remain_size=0;
            original_data.close();
            file_number++;
        }
    }

    //把数据从input_char复制到input_float,并分离两个通道的数据
    //cudaMemcpy(input_char_gpu,input_char_cpu,size*sizeof(signed char),cudaMemcpyHostToDevice);
    //cudaDeviceSynchronize();
} 