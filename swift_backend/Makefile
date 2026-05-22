TARGET = libppv_metal.dylib
SRC = Squaring.swift

# Swift compiler with -O for release optimization. Link against Metal
# and Accelerate frameworks. Emit a dynamic library so ctypes can load it.
$(TARGET): $(SRC)
	swiftc -O -emit-library \
		-framework Metal \
		-framework MetalPerformanceShaders \
		-framework Accelerate \
		-o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.swiftmodule *.swiftdoc

.PHONY: clean
