NVCC ?= nvcc

all: repro workaround

repro: repro.cu
	$(NVCC) -o $@ $<

workaround: workaround.cu
	$(NVCC) -o $@ $<

run: all
	./repro
	@echo
	./workaround

clean:
	rm -f repro workaround

.PHONY: all run clean
