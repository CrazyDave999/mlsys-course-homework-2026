# Makefile for 2D Continuous Cumsum CUDA Benchmark

# Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O1 --std=c++14 -Wno-deprecated-gpu-targets

# Target executable
TARGET = cumsum_benchmark

# Source files
SOURCES = cumsum.cu cumsum_host.cu

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCES)
	@echo "Build complete: $(TARGET)"

# Clean target
clean:
	rm -f $(TARGET)
	@echo "Clean complete"

# Run with default parameters (1024x1024, 100 iterations)
run: $(TARGET)
	./$(TARGET)

# Run with small test parameters (128x128, 10 iterations)
small: $(TARGET)
	./$(TARGET) 128 128 10

# Run with medium test parameters (2048x2048, 50 iterations)
medium: $(TARGET)
	./$(TARGET) 2048 2048 50

# Run with large test parameters (4096x4096, 100 iterations)
large: $(TARGET)
	./$(TARGET) 4096 4096 100

# Run with large test parameters (4096x4096, 100 iterations)
final_test: $(TARGET)
	./$(TARGET) 2048 16384 100

# Help target
help:
	@echo "Available targets:"
	@echo "  make          - Build the benchmark executable"
	@echo "  make clean    - Remove the executable"
	@echo "  make run      - Build and run with default parameters (1024x1024, 100 iters)"
	@echo "  make test     - Build and run with small test (128x128, 10 iters)"
	@echo "  make medium   - Build and run with medium test (2048x2048, 50 iters)"
	@echo "  make large    - Build and run with large test (4096x4096, 100 iters)"
	@echo "  make help     - Show this help message"

.PHONY: all clean run test medium large help
