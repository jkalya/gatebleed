# Define the C compiler and flags
CC = gcc
CFLAGS = -march=native -O0

# Find all the .c files in the current directory
SRC = $(wildcard *.c)

# Generate the corresponding .o files (object files)
OBJ = $(SRC:.c=.o)

# Generate the corresponding executables from the .c files
EXEC = $(SRC:.c=)

# Default target: compile all source files into their corresponding executables
all: $(EXEC)

# Rule to compile each .c file into its corresponding .o file
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to link each .o file into its corresponding executable
%: %.o
	$(CC) $(CFLAGS) $< -o $@

# Clean up objec
