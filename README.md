# Parallel-BMA

This is the source of the expeirments on a paper entitled "Parallel SIMD CPU and GPU Implementations of Berlekamp–Massey Algorithm and Its Error Correction Application" which published in International Journal of Parallel Programming at 2018. If you used this source code, please cite this paper.

The Berlekamp-Massy algorithm finds the shortest linear feedback shift register (LFSR) for a binary input sequence.  A wide range of
applications like cryptography and digital signal processing use the BMA algorithm.  This research uses different parallel mechanisms offered by heterogeneous hardward in order to achieve the best possible performance for BMA.  At first, the bitwise implementation of the BMA algorithm is almost 35 times faster than typical implementation.  The GPU device with thousands of processing cores can bring great speedup over the best CPU implementation.  Two other parallel mechanisms offered by GPU are concurrent kernel execution and streaming.  They achieve 14.5 and 2.2 times of speedup compared to CPU serial and typical CUDA implementations, respectively.  SIMD instructions provide data level parallelism.  A code with SIMD instructions can be 4.6 and 35 times faster than a bitwise CPU and typical implementations, respectively.  A multi-threading OpenMP implementation can use several CPU cores.  The OpenMP code can deliver more than 10 times of speedup.  Also, performance of the openMP code with SIMD instructions is compared with GPU implementation with CUDA streams.  The GPU implementation using the CUDA streams is more than 1.7 times faster than the openMP implementation.  The effectiveness of proposed method is evaluated in an error correction code implementation and it achieves 6.8 times of speedup.

This repository contains source code of different implementaions of BMA algorithm and also the application of this algorihtm in the NAND experiment folder. 

Thanks
