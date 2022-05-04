CC      := g++
CFLAGS  := -std=c++11 -O2 -g -Wall -Wextra -Wshadow -pedantic -I/usr/local/cuda/targets/x86_64-linux/include -I/usr/local/cuda/samples/common/inc

LDFLAGS := -L=/usr/local/cuda/targets/x86_64-linux/lib -lbenchmark -lm

all: cufft-single-benchmark cufft-single-2d-benchmark cufft-single-3d-benchmark

	
cufft-single-benchmark: cufft-single-benchmark.o
	$(CC) -o $@ $^ $(LDFLAGS) -lcufft -lcudart

cufft-single-benchmark.o: cufft-single-benchmark.cc
	$(CC) -c $(CFLAGS) -ICommon/ $<

cufft-single-2d-benchmark: cufft-single-2d-benchmark.o
	$(CC) -o $@ $^ $(LDFLAGS) -lcufft -lcudart

cufft-single-2d-benchmark.o: cufft-single-2d-benchmark.cc
	$(CC) -c $(CFLAGS) -ICommon/ $<

cufft-single-3d-benchmark: cufft-single-3d-benchmark.o
	$(CC) -o $@ $^ $(LDFLAGS) -lcufft -lcudart

cufft-single-3d-benchmark.o: cufft-single-3d-benchmark.cc
	$(CC) -c $(CFLAGS) -ICommon/ $<


.PHONY: clean

clean:
	rm *.o cufft-single-2d-benchmark cufft-single-3d-benchmark cufft-single-benchmark 