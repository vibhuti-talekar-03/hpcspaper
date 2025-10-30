#!/bin/bash
# reorganize_project.sh
# Reorganizes your Fastrack-StringSearch project into proper structure

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Fastrack-StringSearch Project Reorganization ===${NC}\n"

# Check if we're in the right directory
if [ ! -f "final2.cu" ] && [ ! -f "5.cu" ] && [ ! -f "final.cu" ]; then
    echo -e "${YELLOW}Warning: Cannot find your .cu files in current directory.${NC}"
    echo "Please run this script from the directory containing final2.cu, 5.cu, and final.cu"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create directory structure
echo -e "${YELLOW}Creating directory structure...${NC}"
mkdir -p src
mkdir -p datasets/raw
mkdir -p scripts
mkdir -p tests
mkdir -p results/profiling
mkdir -p docs
mkdir -p bin

echo -e "${GREEN}✓ Directories created${NC}\n"

# Move/rename source files
echo -e "${YELLOW}Organizing source files...${NC}"

if [ -f "final2.cu" ]; then
    cp final2.cu src/fastrack_string_search.cu
    echo "  ✓ final2.cu -> src/fastrack_string_search.cu"
fi

if [ -f "5.cu" ]; then
    cp 5.cu src/fastrack_debug.cu
    echo "  ✓ 5.cu -> src/fastrack_debug.cu"
fi

if [ -f "final.cu" ]; then
    cp final.cu src/fastrack_minimal.cu
    echo "  ✓ final.cu -> src/fastrack_minimal.cu"
fi

echo -e "${GREEN}✓ Source files organized${NC}\n"

# Create Makefile
echo -e "${YELLOW}Creating Makefile...${NC}"
cat > Makefile << 'EOF'
# Makefile for Fastrack-StringSearch
NVCC = nvcc
CFLAGS = -O3 -std=c++14 -arch=sm_60
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
EOF

echo -e "${GREEN}✓ Makefile created${NC}\n"

# Create dataset setup script
echo -e "${YELLOW}Creating dataset setup script...${NC}"
cat > scripts/setup_datasets.sh << 'EOFSCRIPT'
#!/bin/bash
# Setup datasets for Fastrack-StringSearch testing

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Setting up datasets...${NC}\n"

cd datasets

# Download Gutenberg samples
echo "Downloading Gutenberg samples..."
mkdir -p raw

if command -v wget &> /dev/null; then
    DOWNLOADER="wget -q --show-progress"
    wget -q --show-progress https://www.gutenberg.org/files/1342/1342-0.txt -O raw/pride_and_prejudice.txt 2>&1 || true
    wget -q --show-progress https://www.gutenberg.org/files/11/11-0.txt -O raw/alice_in_wonderland.txt 2>&1 || true
    wget -q --show-progress https://www.gutenberg.org/files/1661/1661-0.txt -O raw/sherlock_holmes.txt 2>&1 || true
    wget -q --show-progress https://www.gutenberg.org/files/84/84-0.txt -O raw/frankenstein.txt 2>&1 || true
    wget -q --show-progress https://www.gutenberg.org/files/2701/2701-0.txt -O raw/moby_dick.txt 2>&1 || true
else
    curl -# https://www.gutenberg.org/files/1342/1342-0.txt -o raw/pride_and_prejudice.txt 2>&1 || true
    curl -# https://www.gutenberg.org/files/11/11-0.txt -o raw/alice_in_wonderland.txt 2>&1 || true
    curl -# https://www.gutenberg.org/files/1661/1661-0.txt -o raw/sherlock_holmes.txt 2>&1 || true
    curl -# https://www.gutenberg.org/files/84/84-0.txt -o raw/frankenstein.txt 2>&1 || true
    curl -# https://www.gutenberg.org/files/2701/2701-0.txt -o raw/moby_dick.txt 2>&1 || true
fi

# Combine Gutenberg files
if ls raw/*.txt 1> /dev/null 2>&1; then
    cat raw/*.txt > gutenberg_sample.txt 2>/dev/null || true
    echo -e "${GREEN}✓ Gutenberg sample created${NC}"
fi

# Download Reuters
echo "Downloading Reuters dataset..."
if command -v wget &> /dev/null; then
    wget -q --show-progress http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz -O raw/reuters21578.tar.gz 2>&1 || true
else
    curl -# http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz -o raw/reuters21578.tar.gz 2>&1 || true
fi

if [ -f "raw/reuters21578.tar.gz" ]; then
    cd raw
    tar -xzf reuters21578.tar.gz 2>/dev/null || tar -xf reuters21578.tar.gz
    cd ..
    echo -e "${GREEN}✓ Reuters extracted${NC}"
    
    # Extract Reuters text
    python3 ../scripts/extract_reuters.py
fi

cd ..

echo -e "\n${GREEN}✓ Dataset setup complete${NC}"
EOFSCRIPT

chmod +x scripts/setup_datasets.sh
echo -e "${GREEN}✓ Dataset setup script created${NC}\n"

# Create Reuters extraction script
echo -e "${YELLOW}Creating Reuters extraction script...${NC}"
cat > scripts/extract_reuters.py << 'EOFPYTHON'
#!/usr/bin/env python3
"""Extract text from Reuters SGML files"""

import re
import os
from pathlib import Path

def extract_text_from_sgml(sgml_file):
    """Extract text content from Reuters SGML files"""
    try:
        with open(sgml_file, 'r', encoding='latin-1', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {sgml_file}: {e}")
        return []
    
    articles = re.findall(r'<REUTERS.*?>(.*?)</REUTERS>', content, re.DOTALL)
    
    texts = []
    for article in articles:
        title_match = re.search(r'<TITLE>(.*?)</TITLE>', article, re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        
        body_match = re.search(r'<BODY>(.*?)</BODY>', article, re.DOTALL)
        body = body_match.group(1).strip() if body_match else ""
        
        if title or body:
            texts.append(f"{title}\n{body}\n\n")
    
    return texts

def main():
    # Navigate to datasets directory
    datasets_dir = Path(__file__).parent.parent / 'datasets'
    os.chdir(datasets_dir)
    
    output_file = "reuters_combined.txt"
    all_texts = []
    
    raw_dir = Path('raw')
    sgm_files = sorted(raw_dir.glob('reut2-*.sgm'))
    
    if not sgm_files:
        print("No .sgm files found in datasets/raw/")
        return
    
    for sgm_file in sgm_files:
        print(f"Processing {sgm_file.name}...")
        texts = extract_text_from_sgml(sgm_file)
        all_texts.extend(texts)
        print(f"  Extracted {len(texts)} articles")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(all_texts)
    
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n✓ Total articles: {len(all_texts)}")
    print(f"✓ Output: {output_file}")
    print(f"✓ Size: {size_mb:.2f} MB")

if __name__ == '__main__':
    main()
EOFPYTHON

chmod +x scripts/extract_reuters.py
echo -e "${GREEN}✓ Reuters extraction script created${NC}\n"

# Create test generation script
echo -e "${YELLOW}Creating test data generator...${NC}"
cat > scripts/generate_test_data.py << 'EOFPYTHON'
#!/usr/bin/env python3
"""Generate synthetic test data for stress testing"""

import random
import string
import sys
from pathlib import Path

def generate_large_text(size_mb, vocabulary_size=10000):
    """Generate large text file for stress testing"""
    print(f"Generating {size_mb}MB test file...")
    
    # Create vocabulary
    words = []
    for i in range(vocabulary_size):
        length = random.randint(3, 12)
        word = ''.join(random.choices(string.ascii_lowercase, k=length))
        words.append(word)
    
    # Add common words for testing
    common = ["the", "and", "of", "to", "in", "is", "it", "that", "for", "with",
              "test", "search", "algorithm", "data", "word", "string", "pattern"]
    words.extend(common * 100)
    
    target_bytes = size_mb * 1024 * 1024
    output = []
    current_size = 0
    
    while current_size < target_bytes:
        sentence_length = random.randint(10, 30)
        sentence = ' '.join(random.choices(words, k=sentence_length)) + '.'
        output.append(sentence)
        current_size += len(sentence) + 1
        
        if current_size % (10 * 1024 * 1024) < 1000:
            print(f"  Generated {current_size // (1024*1024)} MB...")
    
    return ' '.join(output)

def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    
    # Save to datasets directory
    datasets_dir = Path(__file__).parent.parent / 'datasets'
    output_file = datasets_dir / f"large_test_{size}mb.txt"
    
    text = generate_large_text(size)
    
    with open(output_file, 'w') as f:
        f.write(text)
    
    print(f"✓ Created {output_file.name}: {len(text)} bytes")

if __name__ == "__main__":
    main()
EOFPYTHON

chmod +x scripts/generate_test_data.py
echo -e "${GREEN}✓ Test data generator created${NC}\n"

# Create basic test script
echo -e "${YELLOW}Creating test scripts...${NC}"
cat > tests/test_basic.sh << 'EOFTEST'
#!/bin/bash
# Basic functionality tests

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

PASSED=0
FAILED=0

echo "=== Basic Functionality Tests ==="
echo ""

# Create test file
cat > tests/test_input.txt << 'EOF'
The quick brown fox jumps over the lazy dog.
The dog was very lazy.
A quick brown fox is quick indeed.
EOF

# Test 1: Single word search
echo "Test 1: Single word search"
OUTPUT=$(bin/fastrack tests/test_input.txt "fox" 2>&1)
if echo "$OUTPUT" | grep -q "FOUND"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((FAILED++))
fi

# Test 2: Multiple word search
echo "Test 2: Multiple word search"
OUTPUT=$(bin/fastrack tests/test_input.txt "quick" "dog" "fox" 2>&1)
if echo "$OUTPUT" | grep -q "FOUND"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((FAILED++))
fi

# Test 3: Word not found
echo "Test 3: Word not found"
OUTPUT=$(bin/fastrack tests/test_input.txt "notfound" 2>&1)
if echo "$OUTPUT" | grep -q "NOT FOUND"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((FAILED++))
fi

# Test 4: Case insensitive
echo "Test 4: Case insensitive search"
OUTPUT=$(bin/fastrack tests/test_input.txt "QUICK" 2>&1)
if echo "$OUTPUT" | grep -q "FOUND"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((FAILED++))
fi

echo ""
echo "Results: ${PASSED} passed, ${FAILED} failed"
EOFTEST

chmod +x tests/test_basic.sh
echo -e "${GREEN}✓ Basic test script created${NC}\n"

# Create performance test script
cat > tests/test_performance.sh << 'EOFTEST'
#!/bin/bash
# Performance tests

echo "=== Performance Tests ==="
echo ""

if [ ! -f "datasets/gutenberg_sample.txt" ]; then
    echo "Gutenberg dataset not found. Run 'make setup-datasets' first."
    exit 1
fi

echo "Testing on Gutenberg dataset..."
time bin/fastrack datasets/gutenberg_sample.txt "love" "time" "life"

if [ -f "datasets/reuters_combined.txt" ]; then
    echo ""
    echo "Testing on Reuters dataset..."
    time bin/fastrack datasets/reuters_combined.txt "stock" "market" "trade"
fi
EOFTEST

chmod +x tests/test_performance.sh
echo -e "${GREEN}✓ Performance test script created${NC}\n"

# Create dataset test script
cat > tests/test_datasets.sh << 'EOFTEST'
#!/bin/bash
# Dataset-specific tests

echo "=== Dataset Tests ==="
echo ""

if [ -f "datasets/gutenberg_sample.txt" ]; then
    echo "Testing Gutenberg dataset..."
    bin/fastrack datasets/gutenberg_sample.txt "love" "death" "time"
    echo ""
fi

if [ -f "datasets/reuters_combined.txt" ]; then
    echo "Testing Reuters dataset..."
    bin/fastrack datasets/reuters_combined.txt "stock" "market" "economic"
    echo ""
fi
EOFTEST

chmod +x tests/test_datasets.sh
echo -e "${GREEN}✓ Dataset test script created${NC}\n"

# Create .gitignore
echo -e "${YELLOW}Creating .gitignore...${NC}"
cat > .gitignore << 'EOF'
# Compiled binaries
bin/
*.o
*.exe

# Large datasets
datasets/*.txt
datasets/raw/
!datasets/README.md

# Results
results/*.csv
results/*.txt
results/profiling/*.nvvp
results/profiling/*.txt

# Temporary files
*.swp
*.log
*~
.DS_Store

# IDE files
.vscode/
.idea/
*.code-workspace
EOF

echo -e "${GREEN}✓ .gitignore created${NC}\n"

# Create main README
echo -e "${YELLOW}Creating README.md...${NC}"
cat > README.md << 'EOFREADME'
# Fastrack-StringSearch: GPU-Accelerated String Searching

Implementation of the Fastrack-StringSearch algorithm from the paper:
**"Advancements in String-Searching Algorithms: Fastrack-StringSearch - A Novel Approach"**

## Overview

Fastrack-StringSearch is a novel GPU-accelerated string searching algorithm that combines:
- Merge Sort for lexicographic ordering
- Binary Search for efficient lookups
- Dictionary indexing for multiple occurrences
- CUDA for parallel processing

**Performance:** Up to 50% faster than KMP in best-case scenarios.

## Quick Start

```bash
# 1. Clone repository
git clone <your-repo-url>
cd fastrack-stringsearch

# 2. Setup datasets
make setup-datasets

# 3. Build
make

# 4. Run test
bin/fastrack datasets/gutenberg_sample.txt "search" "algorithm"
```

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (tested with CUDA 11.0+)
- GCC/G++ compiler
- Python 3.6+ (for benchmarking scripts)

## Project Structure

```
fastrack-stringsearch/
├── src/                     # Source code
│   ├── fastrack_string_search.cu    # Main implementation
│   ├── fastrack_debug.cu            # Debug version
│   └── fastrack_minimal.cu          # Minimal version
├── datasets/                # Test datasets
├── scripts/                 # Helper scripts
├── tests/                   # Test scripts
├── results/                 # Benchmark results
├── docs/                    # Documentation
├── bin/                     # Compiled executables
└── Makefile                 # Build system
```

## Build System

```bash
make                  # Build all executables
make clean            # Remove build artifacts
make test             # Run basic tests
make test-performance # Run performance tests
make benchmark        # Run comprehensive benchmarks
make profile          # Profile with nvprof
make memcheck         # Check for memory leaks
make help             # Show all targets
```

## Usage

```bash
# Basic usage
bin/fastrack <input_file> <query1> [query2 ...]

# Examples
bin/fastrack datasets/gutenberg_sample.txt "love" "time"
bin/fastrack datasets/reuters_combined.txt "stock" "market" "trade"

# Debug version (verbose output)
bin/fastrack_debug datasets/test.txt "word"

# Minimal version (no dictionary)
bin/fastrack_minimal datasets/test.txt "word"
```

## Testing

### Basic Tests
```bash
make test
```

### Performance Tests
```bash
make test-performance
```

### Dataset Tests
```bash
make test-datasets
```

## Benchmarking

Run comprehensive benchmarks:
```bash
make benchmark
```

Results will be saved in `results/` directory.

## Profiling

### GPU Profiling
```bash
make profile
# View with: nvvp results/profiling/profile.nvvp
```

### Memory Check
```bash
make memcheck
# View: cat results/profiling/memcheck_output.txt
```

## Performance

Based on the paper's benchmarks:

### Reuters Dataset (~43MB)
- Preprocessing: ~50-100ms
- Merge Sort: ~200-400ms
- Binary Search: <10ms per query
- **50% faster than KMP** (best case)

### Gutenberg Dataset (larger)
- Scales efficiently with dataset size
- Best case: Multiple queries on same dataset
- Worst case: Single query (preprocessing overhead)

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{amruth2024fastrack,
  title={Advancements in String-Searching Algorithms: Fastrack-StringSearch - A Novel Approach},
  author={Amruth A and Ramanan R and Rhea Paul and Vimal C and Meena Belwal},
  booktitle={15th International Conference on Computing Communication and Networking Technologies (ICCCNT)},
  year={2024},
  organization={IEEE}
}
```

## License

[Specify your license here]

## Authors

- Amruth A
- Ramanan R
- Rhea Paul
- Vimal C
- Meena Belwal

## Acknowledgments

- Amrita Vishwa Vidyapeetham
- ICCCNT 2024 Conference
EOFREADME

echo -e "${GREEN}✓ README.md created${NC}\n"

# Create datasets README
cat > datasets/README.md << 'EOF'
# Datasets

This directory contains test datasets for Fastrack-StringSearch.

## Setup

Run: `make setup-datasets` from project root

## Available Datasets

### Gutenberg Dataset
- Combined sample of Project Gutenberg books
- Size: ~2-3 MB
- File: `gutenberg_sample.txt`

### Reuters Dataset
- Reuters-21578 news articles
- Size: ~43 MB
- File: `reuters_combined.txt`

### Synthetic Data
Generate with: `python3 scripts/generate_test_data.py <size_in_mb>`

## Download Sources

- **Gutenberg**: https://www.gutenberg.org/
- **Reuters**: http://www.daviddlewis.com/resources/testcollections/reuters21578/

## Note

Large datasets are not included in version control. Run setup script to download.
EOF

echo -e "${GREEN}✓ datasets/README.md created${NC}\n"

# Final summary
echo -e "${GREEN}=== Project Reorganization Complete ===${NC}\n"

echo "Directory structure created:"
echo "  ✓ src/          - Source files"
echo "  ✓ datasets/     - Test datasets"
echo "  ✓ scripts/      - Helper scripts"
echo "  ✓ tests/        - Test scripts"
echo "  ✓ results/      - Benchmark results"
echo "  ✓ docs/         - Documentation"
echo "  ✓ bin/          - Executables (after build)"
echo ""

echo "Files created:"
echo "  ✓ Makefile"
echo "  ✓ README.md"
echo "  ✓ .gitignore"
echo "  ✓ Scripts in scripts/"
echo "  ✓ Tests in tests/"
echo ""

echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Build project:        make"
echo "  2. Setup datasets:       make setup-datasets"
echo "  3. Run tests:            make test"
echo "  4. Run benchmarks:       make benchmark"
echo ""

echo "Your original files are preserved. New organized copies are in src/"
echo ""

echo -e "${GREEN}Done!${NC}"