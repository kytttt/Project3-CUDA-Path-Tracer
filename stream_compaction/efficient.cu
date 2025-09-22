#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256 
#endif

#ifndef ENABLE_SHARED_MEMORY
#define ENABLE_SHARED_MEMORY 1
#endif


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void UpSweep(int n, int d, int* data) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int step = 1 << (d + 1);
            int count = n / step;
            if (i >= count) return;

            int idx = (i + 1) * step - 1;
            data[idx] += data[idx - (step >> 1)];
        }

        __global__ void DownSweep(int n, int d, int* data) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int step = 1 << (d + 1);
            int count = n / step;
            if (i >= count) return;

            int idx = (i + 1) * step - 1;
            int temp = data[idx - (step >> 1)];
            data[idx - (step >> 1)] = data[idx];
            data[idx] += temp;
        }


        __global__ void shareMemoryScan(int n, const int* idata, int* odata, int* lastElement)
        {
            extern __shared__ int smem[];

            const int i0 = 2 * blockDim.x * blockIdx.x + threadIdx.x;
            const int i1 = 2 * blockDim.x * blockIdx.x + threadIdx.x + blockDim.x;

            smem[threadIdx.x] = (i0 < n) ? idata[i0] : 0;
            smem[threadIdx.x + blockDim.x] = (i1 < n) ? idata[i1] : 0;
            __syncthreads();


            for (int offset = 1; offset < 2 * blockDim.x; offset <<= 1) {
                int idx = ((threadIdx.x + 1) * (offset << 1)) - 1;
                if (idx < 2 * blockDim.x) {
                    smem[idx] += smem[idx - offset];
                }
                __syncthreads();
            }

            int lastEle = smem[2 * blockDim.x - 1];
            if (threadIdx.x == 0) smem[2 * blockDim.x - 1] = 0;
            __syncthreads();


            for (int offset = blockDim.x; offset >= 1; offset >>= 1) {
                int idx = ((threadIdx.x + 1) * (offset << 1)) - 1;
                if (idx < 2 * blockDim.x) {
                    int tmp = smem[idx - offset];
                    smem[idx - offset] = smem[idx];
                    smem[idx] += tmp;
                }
                __syncthreads();
            }

            if (i0 < n) odata[i0] = smem[threadIdx.x];
            if (i1 < n) odata[i1] = smem[threadIdx.x + blockDim.x];


            if (threadIdx.x == 0) lastElement[blockIdx.x] = lastEle;
        }

        __global__ void addLastEle(int n, int* odata, const int* lastElement)
        {

            int add_last = lastElement[blockIdx.x];

            int i0 = 2 * blockDim.x * blockIdx.x + threadIdx.x;
            int i1 = 2 * blockDim.x * blockIdx.x + threadIdx.x + blockDim.x;

            if (i0 < n) odata[i0] += add_last;
            if (i1 < n) odata[i1] += add_last;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            //timer().startGpuTimer();
            // TODO
            if (!ENABLE_SHARED_MEMORY || n <= 2 * BLOCK_SIZE) {

                const int nPow2 = 1 << ilog2ceil(n);

                int* dev_data = nullptr;

                cudaMalloc(&dev_data, nPow2 * sizeof(int));
                cudaMemset(dev_data, 0, nPow2 * sizeof(int));
                cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

                /*timer().startGpuTimer();*/

                for (int d = 0; d < ilog2ceil(nPow2); ++d) {
                    int count = nPow2 >> (d + 1);
                    if (count == 0) break;
                    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    UpSweep <<<gridSize, BLOCK_SIZE >>> (nPow2, d, dev_data);
                    checkCUDAError("UpSweep fail");
                }

                cudaMemset(dev_data + (nPow2 - 1), 0, sizeof(int));

                for (int d = ilog2ceil(nPow2) - 1; d >= 0; --d) {
                    int count = nPow2 >> (d + 1);
                    if (count == 0) continue;
                    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    DownSweep <<<gridSize, BLOCK_SIZE >>> (nPow2, d, dev_data);
                    checkCUDAError("DownSweep fail");
                }

               /* timer().endGpuTimer();*/

                cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

                cudaFree(dev_data);
            }

            else {

                const int tile = 2 * BLOCK_SIZE;
                const int numBlocks = (n + tile - 1) / tile;

                int* dev_in = nullptr, * dev_out = nullptr;
                int* dev_lastEle = nullptr;

                cudaMalloc(&dev_in, n * sizeof(int));
                cudaMalloc(&dev_out, n * sizeof(int));
                cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

                cudaMalloc(&dev_lastEle, numBlocks * sizeof(int));


                timer().startGpuTimer();

                shareMemoryScan <<<numBlocks, BLOCK_SIZE, tile * sizeof(int) >>> (n, dev_in, dev_out, dev_lastEle);
                

                int m = 1 << ilog2ceil(numBlocks);

                int* dev_tmp = nullptr;
                cudaMalloc(&dev_tmp, m * sizeof(int));
                cudaMemset(dev_tmp, 0, m * sizeof(int));
                cudaMemcpy(dev_tmp, dev_lastEle, numBlocks * sizeof(int), cudaMemcpyDeviceToDevice);

                for (int d = 0; d < ilog2ceil(m); ++d) {
                    int count = m >> (d + 1);
                    if (count == 0) break;
                    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    UpSweep <<<gridSize, BLOCK_SIZE >> > (m, d, dev_tmp);
                }

                cudaMemset(dev_tmp + (m - 1), 0, sizeof(int));

                for (int d = ilog2ceil(m) - 1; d >= 0; --d) {
                    int count = m >> (d + 1);
                    if (count == 0) continue;
                    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    DownSweep <<<gridSize, BLOCK_SIZE >>> (m, d, dev_tmp);
                }

                cudaMemcpy(dev_lastEle, dev_tmp, numBlocks * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaFree(dev_tmp);

                addLastEle <<<numBlocks, BLOCK_SIZE >>> (n, dev_out, dev_lastEle);

                timer().endGpuTimer();

                cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);

                cudaFree(dev_in);
                cudaFree(dev_out);
                cudaFree(dev_lastEle);
            }


        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // TODO
			/*const int blockSize = 256;*/
			const int gridSizeN = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

			int* boolArray = nullptr;
			int* indices = nullptr;
			int* dev_data = nullptr;
            int* output = nullptr;

			const int nPow2 = 1 << ilog2ceil(n);

			cudaMalloc(&boolArray, n * sizeof(int));
			cudaMalloc(&indices, nPow2 * sizeof(int));
			cudaMalloc(&dev_data, n * sizeof(int));
            cudaMalloc(&output, n * sizeof(int));

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

			Common::kernMapToBoolean <<<gridSizeN, BLOCK_SIZE >>> (n, boolArray, dev_data);
			checkCUDAError("ToBoolean fail");

			cudaMemset(indices, 0, nPow2 * sizeof(int));
			cudaMemcpy(indices, boolArray, n * sizeof(int), cudaMemcpyDeviceToDevice);

            //upsweep
            for (int d = 0; d < ilog2ceil(nPow2); ++d) {
                int count = nPow2 >> (d + 1);
                if (count == 0) break;
                int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                UpSweep <<<gridSize, BLOCK_SIZE >>> (nPow2, d, indices);
                checkCUDAError("UpSweep fail");
            }

            cudaMemset(indices + (nPow2 - 1), 0, sizeof(int));

            //downsweep
            for (int d = ilog2ceil(nPow2) - 1; d >= 0; --d) {
                int count = nPow2 >> (d + 1);
                if (count == 0) continue;
                int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                DownSweep <<<gridSize, BLOCK_SIZE >>> (nPow2, d, indices);
                checkCUDAError("DownSweep fail");
            }

            int lastIndex = 0, lastBool = 0;
            cudaMemcpy(&lastIndex, indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, boolArray + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + lastBool;

            //scatter
            Common::kernScatter <<<gridSizeN, BLOCK_SIZE >>> (
                n, output, dev_data, boolArray, indices);
            checkCUDAError("Scatter fail");

            timer().endGpuTimer();

            cudaMemcpy(odata, output, count * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_data);
            cudaFree(output);
            cudaFree(boolArray);
            cudaFree(indices);


            return count;
            return -1;
        }
    }
}
