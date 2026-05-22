CC      = clang
CFLAGS  = -O3 -arch arm64 -fPIC -Wall -Wextra -std=c11 -DACCELERATE_NEW_LAPACK
LDFLAGS = -dynamiclib -framework Accelerate

TARGET = libppv_squaring.dylib
SOURCES = squaring.c
HEADERS = squaring.h

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: clean
