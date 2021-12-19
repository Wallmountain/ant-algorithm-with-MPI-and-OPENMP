# ant-algorithm-with-MPI-and-OPENMP
mpicc -g  -o <exec> code.c -fopenmp -lm
mpiexec -np $np ./<exec> "file.txt"
