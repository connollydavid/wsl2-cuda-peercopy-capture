NVCC ?= nvcc

all: demo capturable matrix

demo: demo.cu
	$(NVCC) -o $@ $<

capturable: capturable.cu
	$(NVCC) -o $@ $<

matrix: matrix.cu
	$(NVCC) -o $@ $<

run: all
	./demo
	@echo
	./capturable
	@echo
	./matrix

clean:
	rm -f demo capturable matrix

.PHONY: all run clean
