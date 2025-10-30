// fastrack_debug_full_fixed.cu
// Paper-faithful Fastrack-StringSearch with verbose printing after each step.
// No hashing. GPU Merge Sort + GPU Binary Search on word-start positions.
//
// Compile:
// nvcc -O3 -std=c++14 -arch=sm_60 -o fastrack fastrack_debug_full_fixed.cu
//
// Run:
// ./fastrack input.txt query1 [query2 ...]

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

#define CHECK_CUDA(call) do {                              \
    cudaError_t err = call;                                \
    if (err != cudaSuccess) {                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n",          \
            __FILE__, __LINE__, cudaGetErrorString(err));  \
        exit(EXIT_FAILURE);                                \
    }                                                      \
} while(0)

// print limits (adjust if you want more output)
static const int PRINT_CHARS = 200;
static const int PRINT_WORDS = 128;

// ---------------- Device helpers ----------------
__device__ inline bool isAlnumDev(unsigned char c) {
    return ((c >= '0' && c <= '9') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= 'a' && c <= 'z'));
}

// Compare two words lexicographically: returns -1 if A < B, 0 equal, +1 if A > B
__device__ int cmpWordsAt(const char *d_text, int posA, int posB, int textLen) {
    int a = posA, b = posB;
    while (a < textLen && b < textLen) {
        unsigned char ca = d_text[a];
        unsigned char cb = d_text[b];
        bool aa = isAlnumDev(ca);
        bool bb = isAlnumDev(cb);
        if (!aa && !bb) return 0; // both ended -> equal
        if (!aa) return -1; // A ended, B hasn't -> A < B
        if (!bb) return 1;  // B ended, A hasn't -> A > B
        // normalize to lowercase for comparison
        if (ca >= 'A' && ca <= 'Z') ca = ca - 'A' + 'a';
        if (cb >= 'A' && cb <= 'Z') cb = cb - 'A' + 'a';
        if (ca < cb) return -1;
        if (ca > cb) return 1;
        a++; b++;
    }
    // If one goes out-of-bounds, treat as delimiter
    bool aEnd = (a >= textLen) || !isAlnumDev((unsigned char)d_text[a]);
    bool bEnd = (b >= textLen) || !isAlnumDev((unsigned char)d_text[b]);
    if (aEnd && bEnd) return 0;
    if (aEnd) return -1;
    return 1;
}

// ---------------- Kernels ----------------

// Step 1: flag alnum characters
__global__ void kernel_flagChars(const char *d_text, int *d_flag, int textLen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= textLen) return;
    unsigned char c = d_text[i];
    d_flag[i] = isAlnumDev(c) ? 1 : 0;
}

// Step 2: mark word starts
__global__ void kernel_markStarts(const int *d_flag, int *d_start, int textLen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= textLen) return;
    if (d_flag[i] == 1 && (i == 0 || d_flag[i-1] == 0)) d_start[i] = 1;
    else d_start[i] = 0;
}

// Step 3: scatter positions
__global__ void kernel_scatterPos(const int *d_start, const int *d_prefix, int *d_pos, int textLen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= textLen) return;
    if (d_start[i]) {
        int idx = d_prefix[i];
        d_pos[idx] = i;
    }
}

// Merge step: merge runs of size width; number of items = wordCount
__global__ void kernel_mergeStep(const char *d_text, const int *d_inPos, int *d_outPos, int wordCount, int width, int textLen) {
    int mergeId = blockIdx.x * blockDim.x + threadIdx.x;
    long long start = (long long)mergeId * (2LL * width);
    if (start >= wordCount) return;
    long long mid = min(start + width, (long long)wordCount);
    long long end = min(start + 2LL * width, (long long)wordCount);
    long long i = start, j = mid, k = start;
    while (i < mid && j < end) {
        int pa = d_inPos[i];
        int pb = d_inPos[j];
        int cmp = cmpWordsAt(d_text, pa, pb, textLen);
        if (cmp <= 0) { // left <= right
            d_outPos[k++] = d_inPos[i++];
        } else {
            d_outPos[k++] = d_inPos[j++];
        }
    }
    while (i < mid) { d_outPos[k++] = d_inPos[i++]; }
    while (j < end) { d_outPos[k++] = d_inPos[j++]; }
}

// Kernel to mark dictionary group starts (1 = start of a new unique word)
__global__ void kernel_markGroupStarts(const int *d_sortedPos, int *d_groupStart,
                                       const char *d_text, int textLen, int wordCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= wordCount) return;
    if (idx == 0) { d_groupStart[0] = 1; return; }
    int pcur = d_sortedPos[idx];
    int pprev = d_sortedPos[idx - 1];
    int cmp = cmpWordsAt(d_text, pcur, pprev, textLen);
    d_groupStart[idx] = (cmp == 0) ? 0 : 1; // new group if different
}

// Device string compare for binary search: return 0 equal, -1 if word < query, +1 if >
__device__ int cmpWordWithQuery(const char *d_text, int textLen, int wordPos, const char *q) {
    int a = wordPos;
    int qi = 0;
    while (true) {
        unsigned char cq = (unsigned char)q[qi];
        bool qEnd = (cq == '\0');
        unsigned char ct = (a < textLen) ? (unsigned char)d_text[a] : '\0';
        bool tEnd = (a >= textLen) || !isAlnumDev(ct);
        if (qEnd && tEnd) return 0;
        if (qEnd) return 1;   // query shorter => word > query
        if (tEnd) return -1;  // word ended but query not => word < query
        unsigned char cqnorm = cq;
        if (cqnorm >= 'A' && cqnorm <= 'Z') cqnorm = cqnorm - 'A' + 'a';
        unsigned char ctnorm = ct;
        if (ctnorm >= 'A' && ctnorm <= 'Z') ctnorm = ctnorm - 'A' + 'a';
        if (ctnorm < cqnorm) return -1;
        if (ctnorm > cqnorm) return 1;
        a++; qi++;
    }
    return 0;
}

// Binary search kernel (each thread handles one query). Finds any matching index or -1.
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

// ---------------- Host helpers ----------------
static std::string readFileToString(const char *fname) {
    std::ifstream in(fname, std::ios::binary);
    if (!in.is_open()) { fprintf(stderr, "Cannot open %s\n", fname); exit(1); }
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return s;
}

static void printTextWindow(const std::string &s, int limit = PRINT_CHARS) {
    int m = std::min((int)s.size(), limit);
    printf("=== Text (first %d chars) ===\n", m);
    for (int i = 0; i < m; ++i) putchar(s[i]);
    if ((int)s.size() > m) printf("\n... (total %zu bytes)\n\n", s.size()); else printf("\n\n");
}

static void printIntArrayWindow(const int *h, int n, int limit, const char *title) {
    int m = std::min(n, limit);
    printf("=== %s (first %d entries; total %d) ===\n", title, m, n);
    for (int i = 0; i < m; ++i) printf("%4d: %d\n", i, h[i]);
    if (n > m) printf("... (printed %d of %d)\n", m, n);
    printf("\n");
}

static void printWordsFromPositions(const std::string &text, const std::vector<int> &pos, int limit, const char *title) {
    int m = std::min((int)pos.size(), limit);
    printf("=== %s (first %d words) ===\n", title, m);
    for (int i = 0; i < m; ++i) {
        int p = pos[i];
        if (p < 0 || p >= (int)text.size()) { printf("%4d: pos=%d <oob>\n", i, p); continue; }
        int j = p;
        std::string w;
        while (j < (int)text.size() && isalnum((unsigned char)text[j])) { w.push_back(text[j]); j++; }
        printf("%4d: pos=%6d word=[%s]\n", i, p, w.c_str());
    }
    if ((int)pos.size() > m) printf("... (printed %d of %zu)\n", m, pos.size());
    printf("\n");
}

static void buildQueryBuffer(const std::vector<std::string> &queries, std::string &allQ, std::vector<int> &offsets) {
    allQ.clear(); offsets.clear();
    for (const auto &q : queries) {
        offsets.push_back((int)allQ.size());
        allQ += q;
        allQ.push_back('\0');
    }
}

// ---------------- Main ----------------
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_text_file> <query1> [query2 ...]\n", argv[0]);
        return 1;
    }
    const char *infile = argv[1];
    std::vector<std::string> queries;
    for (int i = 2; i < argc; ++i) queries.emplace_back(argv[i]);

    // Read text
    std::string text = readFileToString(infile);
    int textLen = (int)text.size();
    if (textLen == 0) { printf("Empty input\n"); return 0; }
    // ensure null terminator in host copy that we send to device
    std::string textWithNull = text;
    textWithNull.push_back('\0');
    int textWithNullLen = (int)textWithNull.size();

    printTextWindow(text, 256);

    // Device copy of text
    char *d_text = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_text, (size_t)textWithNullLen * sizeof(char)));
    CHECK_CUDA(cudaMemcpy(d_text, textWithNull.data(), (size_t)textWithNullLen * sizeof(char), cudaMemcpyHostToDevice));

    // Step 1: flag characters
    int *d_flag = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_flag, (size_t)textLen * sizeof(int)));
    int tthreads = 256; int tblocks = (textLen + tthreads - 1) / tthreads;
    kernel_flagChars<<<tblocks, tthreads>>>(d_text, d_flag, textLen);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    // copy small window to host
    int flagPrint = std::min(textLen, PRINT_CHARS);
    std::vector<int> h_flag(flagPrint);
    CHECK_CUDA(cudaMemcpy(h_flag.data(), d_flag, (size_t)flagPrint * sizeof(int), cudaMemcpyDeviceToHost));
    printIntArrayWindow(h_flag.data(), flagPrint, flagPrint, "charFlag (sample)");

    // Step 2: mark word starts
    int *d_start = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_start, (size_t)textLen * sizeof(int)));
    kernel_markStarts<<<tblocks, tthreads>>>(d_flag, d_start, textLen);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    int startPrint = std::min(textLen, PRINT_CHARS);
    std::vector<int> h_start(startPrint);
    CHECK_CUDA(cudaMemcpy(h_start.data(), d_start, (size_t)startPrint * sizeof(int), cudaMemcpyDeviceToHost));
    printIntArrayWindow(h_start.data(), startPrint, startPrint, "startFlag (sample)");

    // Step 3: prefix-sum (exclusive) to enumerate words
    thrust::device_ptr<int> dev_start(d_start);
    thrust::device_vector<int> dev_prefix(textLen);
    thrust::exclusive_scan(dev_start, dev_start + textLen, dev_prefix.begin());
    int wordCount = (int)thrust::reduce(dev_start, dev_start + textLen, 0, thrust::plus<int>());
    printf("Step3: prefix-sum done. Detected wordCount = %d\n\n", wordCount);
    if (wordCount == 0) {
        printf("No words found. Exiting.\n");
        CHECK_CUDA(cudaFree(d_text)); CHECK_CUDA(cudaFree(d_flag)); CHECK_CUDA(cudaFree(d_start));
        return 0;
    }
    int prefixPrint = std::min(textLen, PRINT_CHARS);
    std::vector<int> h_prefix(prefixPrint);
    CHECK_CUDA(cudaMemcpy(h_prefix.data(), thrust::raw_pointer_cast(dev_prefix.data()), (size_t)prefixPrint * sizeof(int), cudaMemcpyDeviceToHost));
    printIntArrayWindow(h_prefix.data(), prefixPrint, prefixPrint, "prefixSum (sample)");

    // Step 4: scatter word positions into compact d_pos[wordCount]
    int *d_pos = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_pos, (size_t)wordCount * sizeof(int)));
    kernel_scatterPos<<<tblocks, tthreads>>>(d_start, thrust::raw_pointer_cast(dev_prefix.data()), d_pos, textLen);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    // Copy some positions to host for printing
    int posPrintCount = std::min(wordCount, PRINT_WORDS);
    std::vector<int> h_pos(posPrintCount);
    CHECK_CUDA(cudaMemcpy(h_pos.data(), d_pos, (size_t)posPrintCount * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> h_pos_all;
    if (wordCount <= 200000) {
        h_pos_all.resize(wordCount);
        CHECK_CUDA(cudaMemcpy(h_pos_all.data(), d_pos, (size_t)wordCount * sizeof(int), cudaMemcpyDeviceToHost));
    }
    printWordsFromPositions(text, h_pos_all.empty() ? h_pos : h_pos_all, PRINT_WORDS, "Scattered word positions (sample)");

    // Step 5: GPU merge sort of positions by lexicographic word order
    printf("Step5: starting GPU merge sort on %d words...\n", wordCount);
    int *d_pos_in = nullptr;
    int *d_pos_out = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_pos_in, (size_t)wordCount * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_pos_out, (size_t)wordCount * sizeof(int)));
    // copy initial positions into d_pos_in (device->device)
    CHECK_CUDA(cudaMemcpy(d_pos_in, d_pos, (size_t)wordCount * sizeof(int), cudaMemcpyDeviceToDevice));
    int mergeThreads = 128;
    for (int width = 1; width < wordCount; width <<= 1) {
        int numMerges = (wordCount + (2 * width - 1)) / (2 * width);
        int blocks = (numMerges + mergeThreads - 1) / mergeThreads;
        kernel_mergeStep<<<blocks, mergeThreads>>>(d_text, d_pos_in, d_pos_out, wordCount, width, textLen);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        std::swap(d_pos_in, d_pos_out);
    }
    // Sorted positions are in d_pos_in
    std::vector<int> h_sortedPreview(posPrintCount);
    CHECK_CUDA(cudaMemcpy(h_sortedPreview.data(), d_pos_in, (size_t)posPrintCount * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> h_sorted_all;
    if (wordCount <= 200000) {
        h_sorted_all.resize(wordCount);
        CHECK_CUDA(cudaMemcpy(h_sorted_all.data(), d_pos_in, (size_t)wordCount * sizeof(int), cudaMemcpyDeviceToHost));
    }
    printWordsFromPositions(text, h_sorted_all.empty() ? h_sortedPreview : h_sorted_all, PRINT_WORDS, "Sorted words (lexicographic) (sample)");
    printf("Step5: GPU merge sort complete.\n\n");

    // Step 6: dictionary indexing: build group-start flags on device
    printf("Step6: Marking dictionary groups on device...\n");
    int *d_groupStart = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_groupStart, (size_t)wordCount * sizeof(int)));
    int gthreads = 256;
    int gblocks = (wordCount + gthreads - 1) / gthreads;
    kernel_markGroupStarts<<<gblocks, gthreads>>>(d_pos_in, d_groupStart, d_text, textLen, wordCount);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<int> h_groupStart(std::min(wordCount, PRINT_WORDS));
    CHECK_CUDA(cudaMemcpy(h_groupStart.data(), d_groupStart, (size_t)h_groupStart.size() * sizeof(int), cudaMemcpyDeviceToHost));
    printIntArrayWindow(h_groupStart.data(), (int)h_groupStart.size(), (int)h_groupStart.size(), "groupStart (sample)");

    // Build a host-side dictionary sample based on the sorted sample (for printing)
    std::vector<int> host_sorted_sample = h_sorted_all.empty() ? h_sortedPreview : h_sorted_all;
    // Create groups on host for the sample only
    std::vector<std::tuple<std::string,int,int>> dictSample;
    for (size_t i = 0; i < host_sorted_sample.size(); ) {
        int p = host_sorted_sample[i];
        std::string w;
        int j = p;
        while (j < (int)text.size() && isalnum((unsigned char)text[j])) { w.push_back(text[j]); j++; }
        int startIdx = (int)i;
        int k = i + 1;
        while (k < (int)host_sorted_sample.size()) {
            int p2 = host_sorted_sample[k];
            std::string w2;
            int j2 = p2;
            while (j2 < (int)text.size() && isalnum((unsigned char)text[j2])) { w2.push_back(text[j2]); j2++; }
            if (w2 == w) k++;
            else break;
        }
        int endIdx = k - 1;
        dictSample.push_back({w, startIdx, endIdx});
        i = k;
        if ((int)dictSample.size() >= 50) break;
    }
    printf("=== Dictionary sample (from sorted sample; up to 50 groups) ===\n");
    for (size_t i = 0; i < dictSample.size(); ++i) {
        auto &t = dictSample[i];
        printf("%3zu: word='%s'  sortedIdxRange=[%d..%d]  count=%d\n",
               i, std::get<0>(t).c_str(), std::get<1>(t), std::get<2>(t),
               std::get<2>(t)-std::get<1>(t)+1);
    }
    printf("\n");

    // Step 7: prepare queries and run binary search kernel
    std::string allQ;
    std::vector<int> qOffsets;
    buildQueryBuffer(queries, allQ, qOffsets);
    int numQ = (int)queries.size();
    char *d_queries = nullptr;
    int *d_qOffsets = nullptr;
    if (allQ.size() == 0) {
        allQ = std::string("\0", 1);
        qOffsets.push_back(0);
        numQ = 1;
    }
    CHECK_CUDA(cudaMalloc((void**)&d_queries, (size_t)allQ.size() * sizeof(char)));
    CHECK_CUDA(cudaMemcpy(d_queries, allQ.data(), (size_t)allQ.size() * sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc((void**)&d_qOffsets, (size_t)numQ * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_qOffsets, qOffsets.data(), (size_t)numQ * sizeof(int), cudaMemcpyHostToDevice));

    int *d_result = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_result, (size_t)numQ * sizeof(int)));

    int qThreads = 128, qBlocks = (numQ + qThreads - 1) / qThreads;
    kernel_binarySearch<<<qBlocks, qThreads>>>(d_text, textLen, d_pos_in, wordCount, d_queries, d_qOffsets, numQ, d_result);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int> h_result(numQ);
    CHECK_CUDA(cudaMemcpy(h_result.data(), d_result, (size_t)numQ * sizeof(int), cudaMemcpyDeviceToHost));

    printf("=== Query Phase Results ===\n");
    for (int qi = 0; qi < numQ; ++qi) {
        int foundIdx = h_result[qi];
        if (foundIdx == -1) {
            printf("Query '%s' NOT FOUND\n", queries[qi].c_str());
        } else {
            int wordPos;
            CHECK_CUDA(cudaMemcpy(&wordPos, d_pos_in + foundIdx, sizeof(int), cudaMemcpyDeviceToHost));
            int j = wordPos;
            std::string w;
            while (j < (int)text.size() && isalnum((unsigned char)text[j])) { w.push_back(text[j]); j++; }
            printf("Query '%s' FOUND at sortedIdx=%d -> pos=%d word='%s'\n", queries[qi].c_str(), foundIdx, wordPos, w.c_str());
            int before = std::max(0, foundIdx - 3);
            int after = std::min(wordCount - 1, foundIdx + 3);
            std::vector<int> nearby(after - before + 1);
            CHECK_CUDA(cudaMemcpy(nearby.data(), d_pos_in + before, (size_t)(after - before + 1) * sizeof(int), cudaMemcpyDeviceToHost));
            printf("  nearby sorted words (idx:pos:word):\n");
            for (int k = 0; k < (int)nearby.size(); ++k) {
                int p = nearby[k];
                std::string ww;
                int jj = p;
                while (jj < (int)text.size() && isalnum((unsigned char)text[jj])) { ww.push_back(text[jj]); jj++; }
                printf("   %4d:%6d:%s\n", before + k, p, ww.c_str());
            }
        }
    }
    printf("\n");

    // Cleanup
    CHECK_CUDA(cudaFree(d_text));
    CHECK_CUDA(cudaFree(d_flag));
    CHECK_CUDA(cudaFree(d_start));
    CHECK_CUDA(cudaFree(d_pos));
    CHECK_CUDA(cudaFree(d_pos_in));
    CHECK_CUDA(cudaFree(d_pos_out));
    CHECK_CUDA(cudaFree(d_groupStart));
    CHECK_CUDA(cudaFree(d_queries));
    CHECK_CUDA(cudaFree(d_qOffsets));
    CHECK_CUDA(cudaFree(d_result));

    printf("Done.\n");
    return 0;
}