# Build everything
make

# Download and setup datasets
make setup-datasets

# Run basic tests
make test

# Run performance tests
make test-performance

# Run on Gutenberg dataset
bin/fastrack datasets/gutenberg_sample.txt "love" "time" "death"

# Run on Reuters dataset
bin/fastrack datasets/reuters_combined.txt "stock" "market" "trade"

# Profile
make profile

# Check memory leaks
make memcheck

# Clean
make clean

# See all options
make help