#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void NaiveScan(int n, int d, int *odata, const int *idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
			int offset = 1 << d;
            if (index >= n) {
                return;
            }
            if (index >= offset) {
                odata[index] = idata[index - offset] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
		}

        __global__ void ToExclusive(int n, int* odata, const int* inclusive) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
            if (index == 0) {
                odata[index] = 0;
                return;
			}
            odata[index] = inclusive[index - 1];
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            /*timer().startGpuTimer();*/
            // TODO
            int* dev_ping = nullptr;
            int* dev_pong = nullptr;

            cudaMalloc(&dev_ping, n * sizeof(int));
            cudaMalloc(&dev_pong, n * sizeof(int));

            cudaMemcpy(dev_ping, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            const int blockSize = 256;
            const int gridSize = (n + blockSize - 1) / blockSize;

            int* src = dev_ping;
            int* dst = dev_pong;

            timer().startGpuTimer();

            for (int d = 0; d < ilog2ceil(n); ++d) {
                NaiveScan <<<gridSize, blockSize >>> (n, d, dst, src);
                checkCUDAError("kernNaive fail");

				std::swap(src, dst);
            }

            ToExclusive <<<gridSize, blockSize >>> (n, dst, src);
            checkCUDAError("Exclusive fail");

            timer().endGpuTimer();

            cudaMemcpy(odata, dst, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy  dst->odata");

            cudaFree(dev_ping);
            cudaFree(dev_pong);
            /*timer().endGpuTimer();*/
        }
    }
}
