#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int sum = 0;
            for (int k = 0; k < n; k++) {
                odata[k] = sum;
                sum += idata[k];
            }
            timer().endCpuTimer();
        }

        void scan_no_timer(int n, int* odata, const int* idata) {

            int sum = 0;
            for (int k = 0; k < n; k++) {
                odata[k] = sum;
                sum += idata[k];
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int count = 0;
            for (int k = 0; k < n; k++) {
                if (idata[k] != 0) {
                    odata[count++] = idata[k];
                }
            }
            timer().endCpuTimer();
			return count;
            return -1;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

            int* bools = new int[n];
            for (int k = 0; k < n; ++k) {
                if (idata[k] == 0) {
                    bools[k] = 0;
                }
                else {
					bools[k] = 1;
                }
            }

            int* indices = new int[n];
            scan_no_timer(n, indices, bools);

            int count = indices[n - 1] + bools[n - 1];

            for (int k = 0; k < n; ++k) {
                if (bools[k]) {
                    odata[indices[k]] = idata[k];
                }
            }

            delete[] bools;
            delete[] indices;
            timer().endCpuTimer();
			return count;
            return -1;
        }
    }
}
