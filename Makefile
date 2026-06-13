NVCC ?= nvcc

all: repro workaround matrix

repro: repro.cu
	$(NVCC) -o $@ $<

workaround: workaround.cu
	$(NVCC) -o $@ $<

matrix: matrix.cu
	$(NVCC) -o $@ $<

run: all
	./repro
	@echo
	./workaround
	@echo
	./matrix

clean:
	rm -f repro workaround matrix

.PHONY: all run clean
