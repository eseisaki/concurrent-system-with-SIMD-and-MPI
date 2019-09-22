all:
	$(MAKE) sequential sse sse_mpi
	@echo "\n\n"
	@./sequential $(N)
	@./sse $(N)
	@lamboot
	@mpiexec -n $(P) ./sse_mpi $(N)

sse_mpi: sse_mpi.o
	mpicc -o sse_mpi sse_mpi.c

sse_mpi.o : sse_mpi.c
	mpicc -c -o sse_mpi.o sse_mpi.c

sequential: sequential.o
	gcc -o sequential sequential.c

sequential.o : sequential.c
	gcc -c -o sequential.o sequential.c

sse: sse.o
	gcc -o sse sse.c -msse4.2

sse.o : sse.c
	gcc -c -o sse.o sse.c -msse4.2

clean:
	rm -f sse sse.o
	rm -f sequential sequential.o
	rm -f sse_mpi sse_mpi.o