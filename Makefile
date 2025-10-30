# Makefile for Fastrack-StringSearch
NVCC = nvcc
CFLAGS = -O3 -std=c++14 -gencode=arch=compute_75,code=sm_75
BINDIR = bin
SRCDIR = src

# Detect compute capability automatically (optional)
# GPU_ARCH = $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
# CFLAGS = -O3 -std=c++14 -arch=sm_$(GPU_ARCH)

all: $(BINDIR)/fastrack $(BINDIR)/fastrack_debug $(BINDIR)/fastrack_minimal

$(BINDIR)/fastrack: $(SRCDIR)/fastrack_string_search.cu | $(BINDIR)
	$(NVCC) $(CFLAGS) -o $@ $<
	@echo "✓ Built main executable: $(BINDIR)/fastrack"

$(BINDIR)/fastrack_debug: $(SRCDIR)/fastrack_debug.cu | $(BINDIR)
	$(NVCC) $(CFLAGS) -o $@ $<
	@echo "✓ Built debug executable: $(BINDIR)/fastrack_debug"

$(BINDIR)/fastrack_minimal: $(SRCDIR)/fastrack_minimal.cu | $(BINDIR)
	$(NVCC) $(CFLAGS) -o $@ $<
	@echo "✓ Built minimal executable: $(BINDIR)/fastrack_minimal"

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -rf $(BINDIR)
	@echo "✓ Cleaned build artifacts"

test: $(BINDIR)/fastrack
	@echo "Running basic tests..."
	bash tests/test_basic.sh

test-performance: $(BINDIR)/fastrack
	@echo "Running performance tests..."
	bash tests/test_performance.sh

test-datasets: $(BINDIR)/fastrack
	@echo "Running dataset tests..."
	bash tests/test_datasets.sh

benchmark: $(BINDIR)/fastrack
	@echo "Running comprehensive benchmarks..."
	python3 scripts/run_benchmarks.py

setup-datasets:
	@echo "Setting up datasets..."
	bash scripts/setup_datasets.sh

profile: $(BINDIR)/fastrack
	@echo "Profiling with nvprof..."
	nvprof -o results/profiling/profile.nvvp $(BINDIR)/fastrack datasets/gutenberg_sample.txt "test"

memcheck: $(BINDIR)/fastrack
	@echo "Checking for memory leaks..."
	cuda-memcheck --leak-check full $(BINDIR)/fastrack datasets/gutenberg_sample.txt "test" > results/profiling/memcheck_output.txt 2>&1

help:
	@echo "Fastrack-StringSearch Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make                  Build all executables"
	@echo "  make clean            Remove build artifacts"
	@echo "  make test             Run basic tests"
	@echo "  make test-performance Run performance tests"
	@echo "  make test-datasets    Run dataset tests"
	@echo "  make benchmark        Run comprehensive benchmarks"
	@echo "  make setup-datasets   Download and setup test datasets"
	@echo "  make profile          Profile with nvprof"
	@echo "  make memcheck         Check for memory leaks"
	@echo "  make help             Show this help message"

.PHONY: all clean test test-performance test-datasets benchmark setup-datasets profile memcheck help
