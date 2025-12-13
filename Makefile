CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra
INC = include
BUILD = build
SRC = src

all: dirs $(BUILD)/utils.o

dirs:
	mkdir -p $(BUILD)

$(BUILD)/utils.o: $(SRC)/utils.c $(INC)/utils.h
	$(CC) $(CFLAGS) -I$(INC) -c $< -o $@

clean:
	rm -rf $(BUILD)
