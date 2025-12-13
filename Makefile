CC=gcc
PYTHON=.venv/bin/python

CFLAGS_BASE=-O3 -Wall -Wextra -Iinclude
CFLAGS_VEC=$(CFLAGS_BASE) -march=native
CFLAGS_NOVEC=$(CFLAGS_BASE) -fno-tree-vectorize
OMPFLAGS=-fopenmp
LDFLAGS=-lm

BUILD=build
SRC=src
INC=include
TESTS=tests

BIN_DENSE=$(BUILD)/dense_gemm
BIN_SPMM=$(BUILD)/spmm_csr
BIN_CORR=$(BUILD)/correctness
BIN_CONV=$(BUILD)/convert_csr
BIN_STREAM=$(BUILD)/stream_bw

BIN_DENSE_VEC=$(BUILD)/dense_gemm_vec
BIN_DENSE_NOVEC=$(BUILD)/dense_gemm_novec
BIN_SPMM_VEC=$(BUILD)/spmm_csr_vec
BIN_SPMM_NOVEC=$(BUILD)/spmm_csr_novec

SRC_UTILS=$(SRC)/utils.c
SRC_DENSE=$(SRC)/dense_gemm.c
SRC_SPMM=$(SRC)/spmm_csr.c
SRC_CONV=$(SRC)/convert_csr.c
SRC_STREAM=$(SRC)/stream_bw.c
SRC_CORR=$(TESTS)/correctness.c

.PHONY: all dirs clean test smoke density threads variants simd working_set cache convert_cost roofline plots

all: dirs $(BIN_DENSE) $(BIN_SPMM) $(BIN_CORR) $(BIN_CONV) $(BIN_STREAM)

dirs:
	mkdir -p $(BUILD) results/raw results/tables results/figures

$(BUILD)/utils.o: $(SRC_UTILS) $(INC)/utils.h
	$(CC) $(CFLAGS_VEC) $(OMPFLAGS) -c $< -o $@

$(BUILD)/dense_gemm.o: $(SRC_DENSE) $(INC)/utils.h
	$(CC) $(CFLAGS_VEC) $(OMPFLAGS) -c $< -o $@

$(BUILD)/spmm_csr.o: $(SRC_SPMM) $(INC)/utils.h
	$(CC) $(CFLAGS_VEC) $(OMPFLAGS) -c $< -o $@

$(BUILD)/convert_csr.o: $(SRC_CONV) $(INC)/utils.h
	$(CC) $(CFLAGS_VEC) $(OMPFLAGS) -c $< -o $@

$(BUILD)/stream_bw.o: $(SRC_STREAM) $(INC)/utils.h
	$(CC) $(CFLAGS_VEC) $(OMPFLAGS) -c $< -o $@

$(BUILD)/correctness.o: $(SRC_CORR) $(INC)/utils.h
	$(CC) $(CFLAGS_VEC) $(OMPFLAGS) -c $< -o $@

$(BIN_DENSE): $(BUILD)/dense_gemm.o $(BUILD)/utils.o
	$(CC) $(CFLAGS_VEC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_SPMM): $(BUILD)/spmm_csr.o $(BUILD)/utils.o
	$(CC) $(CFLAGS_VEC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_CONV): $(BUILD)/convert_csr.o $(BUILD)/utils.o
	$(CC) $(CFLAGS_VEC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_STREAM): $(BUILD)/stream_bw.o $(BUILD)/utils.o
	$(CC) $(CFLAGS_VEC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_CORR): $(BUILD)/correctness.o $(BUILD)/utils.o
	$(CC) $(CFLAGS_VEC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

variants: dirs $(BIN_DENSE_VEC) $(BIN_DENSE_NOVEC) $(BIN_SPMM_VEC) $(BIN_SPMM_NOVEC)

$(BUILD)/utils_vec.o: $(SRC_UTILS) $(INC)/utils.h
	$(CC) $(CFLAGS_VEC) $(OMPFLAGS) -c $< -o $@

$(BUILD)/utils_novec.o: $(SRC_UTILS) $(INC)/utils.h
	$(CC) $(CFLAGS_NOVEC) $(OMPFLAGS) -c $< -o $@

$(BUILD)/dense_gemm_vec.o: $(SRC_DENSE) $(INC)/utils.h
	$(CC) $(CFLAGS_VEC) $(OMPFLAGS) -c $< -o $@

$(BUILD)/dense_gemm_novec.o: $(SRC_DENSE) $(INC)/utils.h
	$(CC) $(CFLAGS_NOVEC) $(OMPFLAGS) -c $< -o $@

$(BUILD)/spmm_csr_vec.o: $(SRC_SPMM) $(INC)/utils.h
	$(CC) $(CFLAGS_VEC) $(OMPFLAGS) -c $< -o $@

$(BUILD)/spmm_csr_novec.o: $(SRC_SPMM) $(INC)/utils.h
	$(CC) $(CFLAGS_NOVEC) $(OMPFLAGS) -c $< -o $@

$(BIN_DENSE_VEC): $(BUILD)/dense_gemm_vec.o $(BUILD)/utils_vec.o
	$(CC) $(CFLAGS_VEC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DENSE_NOVEC): $(BUILD)/dense_gemm_novec.o $(BUILD)/utils_novec.o
	$(CC) $(CFLAGS_NOVEC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_SPMM_VEC): $(BUILD)/spmm_csr_vec.o $(BUILD)/utils_vec.o
	$(CC) $(CFLAGS_VEC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_SPMM_NOVEC): $(BUILD)/spmm_csr_novec.o $(BUILD)/utils_novec.o
	$(CC) $(CFLAGS_NOVEC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

test: $(BIN_CORR)
	./$(BIN_CORR)

smoke: $(BIN_DENSE) $(BIN_SPMM)
	./$(BIN_DENSE) 256 256 256 3 1 128 1
	./$(BIN_SPMM)  256 256 256 0.05 3 1 1

density: $(BIN_DENSE) $(BIN_SPMM)
	./scripts/sweep_density.sh 1024 1024 1024 5 1 128 1

threads: $(BIN_DENSE) $(BIN_SPMM)
	./scripts/sweep_threads.sh 1024 1024 1024 0.05 5 128 1

simd: variants
	./scripts/simd_vs_threads.sh 1024 1024 1024 0.05 5 128 1 10

working_set: $(BIN_DENSE) $(BIN_SPMM)
	./scripts/sweep_size.sh

cache: working_set plots
	@echo "cache done"

convert_cost: $(BIN_CONV) $(BIN_SPMM) dirs
	./scripts/sweep_convert.sh 1024 1024 1024 10 5 128 1
	$(PYTHON) python/plot_convert_cost.py
	@echo "convert cost done"

roofline: $(BIN_STREAM) $(BIN_DENSE) $(BIN_SPMM) dirs
	./scripts/sweep_roofline.sh
	$(PYTHON) python/plot_roofline.py
	@echo "roofline done"

plots:
	$(PYTHON) python/plot_break_even.py
	$(PYTHON) python/plot_speedup.py
	$(PYTHON) python/plot_simd_vs_threads.py
	$(PYTHON) python/plot_working_set.py
	$(PYTHON) python/plot_convert_cost.py
	$(PYTHON) python/plot_roofline.py

clean:
	rm -rf $(BUILD)
