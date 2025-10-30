// fastrack_final.cu
// Paper-faithful Fastrack-StringSearch (no hashing, minimal prints)
// Measures GPU time for each stage.
//
// Compile:
//   nvcc -O3 -std=c++14 -arch=sm_60 -o fastrack_final fastrack_final.cu
// Run:
//   ./fastrack_final input.txt query1 [query2 ...]

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <chrono>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ---------- Device Helpers ----------
__device__ inline bool isAlnumDev(unsigned char c) {
    return ((c >= '0' && c <= '9') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= 'a' && c <= 'z'));
}

// Compare two words lexicographically
__device__ int cmpWordsAt(const char *d_text, int posA, int posB, int textLen) {
    int a = posA, b = posB;
    while (a < textLen && b < textLen) {
        unsigned char ca = d_text[a];
        unsigned char cb = d_text[b];
        bool aa = isAlnumDev(ca);
        bool bb = isAlnumDev(cb);
        if (!aa && !bb) return 0;
        if (!aa) return -1;
        if (!bb) return 1;
        if (ca >= 'A' && ca <= 'Z') ca = ca - 'A' + 'a';
        if (cb >= 'A' && cb <= 'Z') cb = cb - 'A' + 'a';
        if (ca < cb) return -1;
        if (ca > cb) return 1;
        a++; b++;
    }
    bool aEnd = (a >= textLen) || !isAlnumDev((unsigned char)d_text[a]);
    bool bEnd = (b >= textLen) || !isAlnumDev((unsigned char)d_text[b]);
    if (aEnd && bEnd) return 0;
    if (aEnd) return -1;
    return 1;
}

// ---------- Kernels ----------
__global__ void kernel_flagChars(const char *d_text, int *d_flag, int textLen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < textLen) {
        unsigned char c = d_text[i];
        d_flag[i] = isAlnumDev(c) ? 1 : 0;
    }
}

__global__ void kernel_markStarts(const int *d_flag, int *d_start, int textLen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < textLen) {
        d_start[i] = (d_flag[i] == 1 && (i == 0 || d_flag[i-1] == 0)) ? 1 : 0;
    }
}

__global__ void kernel_scatterPos(const int *d_start, const int *d_prefix, int *d_pos, int textLen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < textLen && d_start[i]) {
        int idx = d_prefix[i];
        d_pos[idx] = i;
    }
}

__global__ void kernel_mergeStep(const char *d_text, const int *d_inPos, int *d_outPos, int wordCount, int width, int textLen) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    long long start = (long long)id * (2LL * width);
    if (start >= wordCount) return;
    long long mid = min(start + width, (long long)wordCount);
    long long end = min(start + 2LL * width, (long long)wordCount);
    long long i = start, j = mid, k = start;
    while (i < mid && j < end) {
        int pa = d_inPos[i];
        int pb = d_inPos[j];
        int cmp = cmpWordsAt(d_text, pa, pb, textLen);
        if (cmp <= 0) d_outPos[k++] = d_inPos[i++];
        else d_outPos[k++] = d_inPos[j++];
    }
    while (i < mid) d_outPos[k++] = d_inPos[i++];
    while (j < end) d_outPos[k++] = d_inPos[j++];
}

__device__ int cmpWordWithQuery(const char *d_text, int textLen, int wordPos, const char *q) {
    int a = wordPos, qi = 0;
    while (true) {
        unsigned char cq = (unsigned char)q[qi];
        bool qEnd = (cq == '\0');
        unsigned char ct = (a < textLen) ? (unsigned char)d_text[a] : '\0';
        bool tEnd = (a >= textLen) || !isAlnumDev(ct);
        if (qEnd && tEnd) return 0;
        if (qEnd) return 1;
        if (tEnd) return -1;
        unsigned char cqnorm = (cq >= 'A' && cq <= 'Z') ? cq - 'A' + 'a' : cq;
        unsigned char ctnorm = (ct >= 'A' && ct <= 'Z') ? ct - 'A' + 'a' : ct;
        if (ctnorm < cqnorm) return -1;
        if (ctnorm > cqnorm) return 1;
        a++; qi++;
    }
    return 0;
}

__global__ void kernel_binarySearch(const char *d_text, int textLen, const int *d_sortedPos, int wordCount,
                                    const char *d_queries, const int *d_qOffsets, int numQ, int *d_result) {
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= numQ) return;
    const char *q = &d_queries[d_qOffsets[qid]];
    int lo = 0, hi = wordCount - 1;
    int found = -1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int pos = d_sortedPos[mid];
        int cmp = cmpWordWithQuery(d_text, textLen, pos, q);
        if (cmp == 0) { found = mid; break; }
        else if (cmp < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    d_result[qid] = found;
}

// ---------- Host Utilities ----------
static std::string readFileToString(const char *fname) {
    std::ifstream in(fname, std::ios::binary);
    if (!in.is_open()) { fprintf(stderr, "Cannot open %s\n", fname); exit(1); }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static void buildQueryBuffer(const std::vector<std::string> &queries, std::string &allQ, std::vector<int> &offsets) {
    allQ.clear(); offsets.clear();
    for (const auto &q : queries) {
        offsets.push_back((int)allQ.size());
        allQ += q; allQ.push_back('\0');
    }
}

// ---------- Main ----------
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_text_file> <query1> [query2 ...]\n", argv[0]);
        return 1;
    }
    const char *infile = argv[1];
    std::vector<std::string> queries(argv + 2, argv + argc);

    std::string text = readFileToString(infile);
    if (text.empty()) { printf("Empty file.\n"); return 0; }
    int textLen = text.size();
    text.push_back('\0');

    printf("Loaded text: %d characters\n", textLen);

    // Timing setup
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float ms;

    char *d_text;
    CHECK_CUDA(cudaMalloc(&d_text, text.size()));
    CHECK_CUDA(cudaMemcpy(d_text, text.data(), text.size(), cudaMemcpyHostToDevice));

    int *d_flag, *d_start;
    CHECK_CUDA(cudaMalloc(&d_flag, textLen * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_start, textLen * sizeof(int)));

    int threads = 256;
    int blocks = (textLen + threads - 1) / threads;

    CHECK_CUDA(cudaEventRecord(start));
    kernel_flagChars<<<blocks, threads>>>(d_text, d_flag, textLen);
    kernel_markStarts<<<blocks, threads>>>(d_flag, d_start, textLen);
    CHECK_CUDA(cudaDeviceSynchronize());

    thrust::device_ptr<int> dev_start(d_start);
    thrust::device_vector<int> dev_prefix(textLen);
    thrust::exclusive_scan(dev_start, dev_start + textLen, dev_prefix.begin());
    int wordCount = thrust::reduce(dev_start, dev_start + textLen, 0, thrust::plus<int>());
    CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("Preprocessing complete: %d words detected (%.3f ms)\n", wordCount, ms);

    int *d_pos;
    CHECK_CUDA(cudaMalloc(&d_pos, wordCount * sizeof(int)));
    kernel_scatterPos<<<blocks, threads>>>(d_start, thrust::raw_pointer_cast(dev_prefix.data()), d_pos, textLen);
    CHECK_CUDA(cudaDeviceSynchronize());

    // GPU Merge Sort
    CHECK_CUDA(cudaEventRecord(start));
    int *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, wordCount * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_out, wordCount * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_in, d_pos, wordCount * sizeof(int), cudaMemcpyDeviceToDevice));

    int mthreads = 128;
    for (int width = 1; width < wordCount; width <<= 1) {
        int numMerges = (wordCount + (2 * width - 1)) / (2 * width);
        int mblocks = (numMerges + mthreads - 1) / mthreads;
        kernel_mergeStep<<<mblocks, mthreads>>>(d_text, d_in, d_out, wordCount, width, textLen);
        CHECK_CUDA(cudaDeviceSynchronize());
        std::swap(d_in, d_out);
    }
    CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("GPU Merge Sort complete (%.3f ms)\n", ms);

    // Prepare queries
    std::string allQ; std::vector<int> qOffsets;
    buildQueryBuffer(queries, allQ, qOffsets);
    char *d_queries; int *d_qOffsets; int *d_result;
    CHECK_CUDA(cudaMalloc(&d_queries, allQ.size()));
    CHECK_CUDA(cudaMalloc(&d_qOffsets, qOffsets.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_result, qOffsets.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_queries, allQ.data(), allQ.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_qOffsets, qOffsets.data(), qOffsets.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Binary Search
    CHECK_CUDA(cudaEventRecord(start));
    int qThreads = 128, qBlocks = (queries.size() + qThreads - 1) / qThreads;
    kernel_binarySearch<<<qBlocks, qThreads>>>(d_text, textLen, d_in, wordCount, d_queries, d_qOffsets, queries.size(), d_result);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("Binary Search complete (%.3f ms)\n", ms);

    // Results
    std::vector<int> h_result(queries.size());
    CHECK_CUDA(cudaMemcpy(h_result.data(), d_result, h_result.size() * sizeof(int), cudaMemcpyDeviceToHost));

    printf("=== Query Results ===\n");
    for (size_t i = 0; i < queries.size(); ++i) {
        if (h_result[i] == -1) {
            printf("Query '%s' NOT FOUND\n", queries[i].c_str());
        } else {
            int pos;
            CHECK_CUDA(cudaMemcpy(&pos, d_in + h_result[i], sizeof(int), cudaMemcpyDeviceToHost));
            printf("Query '%s' FOUND at pos=%d (sortedIdx=%d)\n", queries[i].c_str(), pos, h_result[i]);
        }
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_text));
    CHECK_CUDA(cudaFree(d_flag));
    CHECK_CUDA(cudaFree(d_start));
    CHECK_CUDA(cudaFree(d_pos));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_queries));
    CHECK_CUDA(cudaFree(d_qOffsets));
    CHECK_CUDA(cudaFree(d_result));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("Done.\n");
    return 0;
}