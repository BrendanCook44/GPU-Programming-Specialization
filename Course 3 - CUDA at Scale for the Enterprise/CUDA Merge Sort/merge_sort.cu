#include "merge_sort.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
// Based on https://github.com/kevin-albert/cuda-mergesort/blob/master/mergesort.cu

__host__ std::tuple<dim3, dim3, int> parseCommandLineArguments(int argc, char** argv) 
{
    int numElements = 32;
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] && !argv[i][2]) {
            char arg = argv[i][1];
            unsigned int* toSet = 0;
            switch(arg) {
                case 'x':
                    toSet = &threadsPerBlock.x;
                    break;
                case 'y':
                    toSet = &threadsPerBlock.y;
                    break;
                case 'z':
                    toSet = &threadsPerBlock.z;
                    break;
                case 'X':
                    toSet = &blocksPerGrid.x;
                    break;
                case 'Y':
                    toSet = &blocksPerGrid.y;
                    break;
                case 'Z':
                    toSet = &blocksPerGrid.z;
                    break;
                case 'n':
                    i++;
                    numElements = stoi(argv[i]);
                    break;
            }
            if (toSet) {
                i++;
                *toSet = (unsigned int) strtol(argv[i], 0, 10);
            }
        }
    }
    return std::make_tuple(threadsPerBlock, blocksPerGrid, numElements);
}

__host__ long *generateRandomLongArray(int numElements)
{
    long *randomLongs = (long*)malloc(numElements * sizeof(long));

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<long> distribution(LONG_MIN, LONG_MAX);

    for (int i = 0; i < numElements; i++)
    {
        randomLongs[i] = distribution(gen);
    }

    return randomLongs;
}

__host__ void printHostMemory(long *host_mem, int num_elments)
{
    for(int i = 0; i < num_elments; i++)
    {
        printf("%ld ",host_mem[i]);
    }
    printf("\n");
}

__host__ int main(int argc, char** argv) 
{
    dim3 threadsPerBlock, blocksPerGrid;
    int numElements;
    std::tie(threadsPerBlock, blocksPerGrid, numElements) = parseCommandLineArguments(argc, argv);

    long *data = generateRandomLongArray(numElements);

    printf("Unsorted data: ");
    printHostMemory(data, numElements);

    data = mergesort(data, numElements, threadsPerBlock, blocksPerGrid);

    printf("Sorted data: ");
    printHostMemory(data, numElements);
    
    free(data);
    return 0;
}

__host__ void allocateMemory(int numElements, long** D_data, long** D_swp, dim3** D_threads, dim3** D_blocks)
{
    cudaMalloc((void**)D_data, sizeof(long) * numElements);
    cudaMalloc((void**)D_swp, sizeof(long) * numElements);
    cudaMalloc((void**)D_threads, sizeof(dim3));
    cudaMalloc((void**)D_blocks, sizeof(dim3));
}

__host__ long* mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    long* D_data;
    long* D_swp;
    dim3* D_threads;
    dim3* D_blocks;
    
    allocateMemory(size, &D_data, &D_swp, &D_threads, &D_blocks);

    long* A = D_data;
    long* B = D_swp;
    
    cudaMemcpy(D_data, data, sizeof(long) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event);

    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);

        long* temp = A;
        A = B;
        B = temp;
    }

    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    printf("Kernel execution time: %f ms\n", milliseconds);

    cudaMemcpy(data, A, sizeof(long) * size, cudaMemcpyDeviceToHost);

    cudaFree(D_data);
    cudaFree(D_swp);
    cudaFree(D_threads);
    cudaFree(D_blocks);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return data;
}

__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    long start = width * idx * slices;
    long middle;
    long end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
        {
            break;
        }

        middle = MIN(start + (width / 2), size);
        end = MIN(start + width, size);
       
        gpu_bottomUpMerge(source, dest, start, middle, end);
        
        start += width;
    }
}

__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;

    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}