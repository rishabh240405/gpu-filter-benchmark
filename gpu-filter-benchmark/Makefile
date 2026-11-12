NVCC = nvcc
NVCCFLAGS = -std=c++17 -O2
TARGET = gpu_filter_benchmark.exe

all: $(TARGET)

$(TARGET): src/main.cu
	$(NVCC) $(NVCCFLAGS) src/main.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
