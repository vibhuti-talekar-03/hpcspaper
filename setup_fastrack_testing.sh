#!/bin/bash
# setup_fastrack_testing.sh
# Complete setup script for Fastrack-StringSearch testing

set -e  # Exit on error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Fastrack-StringSearch Setup Script ===${NC}\n"

# Check prerequisites
echo "Checking prerequisites..."

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: nvcc not found. Please install CUDA toolkit.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ CUDA found: $(nvcc --version | grep release | awk '{print $5}')${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python3 found${NC}"

# Check wget or curl
if command -v wget &> /dev/null; then
    DOWNLOADER="wget"
elif command -v curl &> /dev/null; then
    DOWNLOADER="curl -O"
else
    echo -e "${RED}Error: Neither wget nor curl found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Downloader: ${DOWNLOADER%%\ *}${NC}\n"

# Create directory structure
echo "Creating directory structure..."
mkdir -p datasets
mkdir -p test_results
mkdir -p scripts
echo -e "${GREEN}✓ Directories created${NC}\n"

# Download Gutenberg samples
echo -e "${YELLOW}Downloading Gutenberg samples...${NC}"
cd datasets

if [ "$DOWNLOADER" = "wget" ]; then
    wget -q --show-progress https://www.gutenberg.org/files/1342/1342-0.txt -O pride_and_prejudice.txt 2>&1 || echo "Skipping..."
    wget -q --show-progress https://www.gutenberg.org/files/11/11-0.txt -O alice_in_wonderland.txt 2>&1 || echo "Skipping..."
    wget -q --show-progress https://www.gutenberg.org/files/1661/1661-0.txt -O sherlock_holmes.txt 2>&1 || echo "Skipping..."
    wget -q --show-progress https://www.gutenberg.org/files/84/84-0.txt -O frankenstein.txt 2>&1 || echo "Skipping..."
    wget -q --show-progress https://www.gutenberg.org/files/2701/2701-0.txt -O moby_dick.txt 2>&1 || echo "Skipping..."
else
    curl -# https://www.gutenberg.org/files/1342/1342-0.txt -o pride_and_prejudice.txt 2>&1 || echo "Skipping..."
    curl -# https://www.gutenberg.org/files/11/11-0.txt -o alice_in_wonderland.txt 2>&1 || echo "Skipping..."
    curl -# https://www.gutenberg.org/files/1661/1661-0.txt -o sherlock_holmes.txt 2>&1 || echo "Skipping..."
    curl -# https://www.gutenberg.org/files/84/84-0.txt -o frankenstein.txt 2>&1 || echo "Skipping..."
    curl -# https://www.gutenberg.org/files/2701/2701-0.txt -o moby_dick.txt 2>&1 || echo "Skipping..."
fi

# Combine Gutenberg files
if ls *.txt 1> /dev/null 2>&1; then
    cat *.txt > gutenberg_sample.txt 2>/dev/null || true
    SIZE=$(du -h gutenberg_sample.txt 2>/dev/null | cut -f1)
    echo -e "${GREEN}✓ Gutenberg sample created: ${SIZE}${NC}"
else
    echo -e "${YELLOW}⚠ No Gutenberg files downloaded (network issue?)${NC}"
fi

# Download Reuters dataset
echo -e "\n${YELLOW}Downloading Reuters dataset...${NC}"
if [ "$DOWNLOADER" = "wget" ]; then
    wget -q --show-progress http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz 2>&1 || echo "Skipping Reuters..."
else
    curl -# http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz -o reuters21578.tar.gz 2>&1 || echo "Skipping Reuters..."
fi

# Extract Reuters
if [ -f "reuters21578.tar.gz" ]; then
    echo "Extracting Reuters..."
    tar -xzf reuters21578.tar.gz 2>/dev/null || tar -xf reuters21578.tar.gz
    echo -e "${GREEN}✓ Reuters extracted${NC}"
    
    # Create extraction script
    cat > ../scripts/extract_reuters.py << 'EOF'
#!/usr/bin/env python3
import re
import os
from pathlib import Path
import sys

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

# Change to datasets directory
os.chdir(os.path.dirname(__file__) or '.')
os.chdir('../datasets')

output_file = "reuters_combined.txt"
all_texts = []

sgm_files = sorted(Path('.').glob('reut2-*.sgm'))
if not sgm_files:
    print("No .sgm files found!")
    sys.exit(1)

for sgm_file in sgm_files:
    print(f"Processing {sgm_file}...")
    texts = extract_text_from_sgml(sgm_file)
    all_texts.extend(texts)
    print(f"  Extracted {len(texts)} articles")

with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(all_texts)

print(f"\nTotal articles: {len(all_texts)}")
print(f"Output: {output_file}")
print(f"Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
EOF
    
    chmod +x ../scripts/extract_reuters.py
    echo "Processing Reuters SGML files..."
    python3 ../scripts/extract_reuters.py
    
    if [ -f "reuters_combined.txt" ]; then
        SIZE=$(du -h reuters_combined.txt | cut -f1)
        echo -e "${GREEN}✓ Reuters combined: ${SIZE}${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Reuters dataset not downloaded (network issue?)${NC}"
fi

cd ..

# Create test generator script
echo -e "\n${YELLOW}Creating test generator...${NC}"
cat > scripts/generate_test_data.py << 'EOF'
#!/usr/bin/env python3
import random
import string
import sys

def generate_large_text(size_mb, vocabulary_size=10000):
    """Generate large text file for stress testing"""
    print(f"Generating {size_mb}MB test file...")
    
    # Create vocabulary
    words = []
    for i in range(vocabulary_size):
        length = random.randint(3, 12)
        word = ''.join(random.choices(string.ascii_lowercase, k=length))
        words.append(word)
    
    # Add common words
    common = ["the", "and", "of", "to", "in", "is", "it", "that", "for", "with",
              "test", "search", "algorithm", "data", "word"]
    words.extend(common * 100)
    
    target_bytes = size_mb * 1024 * 1024
    output = []
    current_size = 0
    
    while current_size < target_bytes:
        sentence_length = random.randint(10, 30)
        sentence = ' '.join(random.choices(words, k=sentence_length)) + '.'
        output.append(sentence)
        current_size += len(sentence) + 1
        
        if current_size % (10 * 1024 * 1024) == 0:
            print(f"  Generated {current_size // (1024*1024)} MB...")
    
    return ' '.join(output)

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    output_file = f"large_test_{size}mb.txt"
    
    text = generate_large_text(size)
    with open(output_file, 'w') as f:
        f.write(text)
    
    print(f"✓ Created {output_file}: {len(text)} bytes")
EOF

chmod +x scripts/generate_test_data.py

# Compile the code
echo -e "\n${YELLOW}Compiling Fastrack-StringSearch...${NC}"
if [ -f "final2.cu" ]; then
    nvcc -O3 -std=c++14 -gencode=arch=compute_75,code=sm_75 -o fastrack final2.cu 2>&1 | grep -i error || true
    if [ -f "fastrack" ]; then
        echo -e "${GREEN}✓ Compilation successful${NC}"
    else
        echo -e "${RED}✗ Compilation failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ final2.cu not found. Please place it in current directory.${NC}"
fi

# Create quick test
echo -e "\n${YELLOW}Creating quick test...${NC}"
cat > quick_test.txt << 'EOF'
The quick brown fox jumps over the lazy dog.
The dog was very lazy and slept all day.
A quick brown fox is very quick indeed.
EOF

# Run quick test
if [ -f "fastrack" ]; then
    echo -e "\n${YELLOW}Running quick test...${NC}"
    ./fastrack quick_test.txt "quick" "dog" "fox"
    echo ""
fi

# Create README
cat > TESTING_README.md << 'EOF'
# Fastrack-StringSearch Testing Guide

## Quick Start

```bash
# Basic test
./fastrack quick_test.txt "word"

# Test on datasets
./fastrack datasets/gutenberg_sample.txt "love" "time" "life"
./fastrack datasets/reuters_combined.txt "stock" "market" "trade"

# Generate large test files
python3 scripts/generate_test_data.py 10  # 10MB file
python3 scripts/generate_test_data.py 100 # 100MB file

# Test on large files
./fastrack large_test_10mb.txt "test" "search"
```

## Performance Profiling

```bash
# Profile with nvprof
nvprof ./fastrack datasets/reuters_combined.txt "stock"

# Check for memory leaks
cuda-memcheck --leak-check full ./fastrack quick_test.txt "test"

# Monitor GPU usage
nvidia-smi -l 1  # Run in separate terminal
```

## Available Datasets

- `datasets/gutenberg_sample.txt` - Combined Gutenberg books
- `datasets/reuters_combined.txt` - Reuters news articles
- Individual Gutenberg books in datasets/

## Test Queries

### Common words (worst case)
```bash
./fastrack datasets/gutenberg_sample.txt "the" "and" "of"
```

### Specific terms (best case)
```bash
./fastrack datasets/gutenberg_sample.txt "Elizabeth" "Darcy"
./fastrack datasets/reuters_combined.txt "stocks" "economic"
```

## Benchmarking

Compare with baseline:
```bash
# Your implementation
time ./fastrack datasets/reuters_combined.txt "stock" "market"

# Python baseline (for comparison)
time grep -o "stock" datasets/reuters_combined.txt | wc -l
```

## Directory Structure

```
.
├── fastrack              # Compiled executable
├── final2.cu             # Source code
├── datasets/             # Test datasets
│   ├── gutenberg_sample.txt
│   ├── reuters_combined.txt
│   └── *.txt (individual books)
├── scripts/              # Helper scripts
│   ├── extract_reuters.py
│   └── generate_test_data.py
├── test_results/         # Results from test runs
└── quick_test.txt        # Small test file
```
EOF

# Final summary
echo -e "\n${GREEN}=== Setup Complete ===${NC}\n"
echo "Summary:"
echo "  ✓ Directories created"

if [ -f "fastrack" ]; then
    echo "  ✓ Fastrack compiled"
else
    echo "  ⚠ Fastrack not compiled (place final2.cu here and rerun)"
fi

if [ -f "datasets/gutenberg_sample.txt" ]; then
    SIZE=$(du -h datasets/gutenberg_sample.txt | cut -f1)
    echo "  ✓ Gutenberg dataset ready (${SIZE})"
else
    echo "  ⚠ Gutenberg dataset not available"
fi

if [ -f "datasets/reuters_combined.txt" ]; then
    SIZE=$(du -h datasets/reuters_combined.txt | cut -f1)
    echo "  ✓ Reuters dataset ready (${SIZE})"
else
    echo "  ⚠ Reuters dataset not available"
fi

echo ""
echo "Next steps:"
echo "  1. Run quick test:    ./fastrack quick_test.txt \"quick\" \"dog\""
echo "  2. Test on Gutenberg: ./fastrack datasets/gutenberg_sample.txt \"love\" \"time\""
echo "  3. Test on Reuters:   ./fastrack datasets/reuters_combined.txt \"stock\" \"market\""
echo "  4. Generate large:    python3 scripts/generate_test_data.py 100"
echo "  5. Profile:           nvprof ./fastrack quick_test.txt \"test\""
echo ""
echo "Read TESTING_README.md for detailed instructions."
echo -e "${GREEN}Happy testing!${NC}\n"
