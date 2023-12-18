#include </home/vboxuser/parallel_project/Matrix.h>
#include <mpi.h>
#include <math.h>

// allocates matrix[rows][cols] 
int allocMatrix(int*** matrix, int rows, int cols) {
	int* p = (int*)malloc(sizeof(int*) * rows * cols);
	if (!p) {
		return -1;
	}
	*matrix = (int**)malloc(rows * sizeof(int*));
	if (!matrix) {
		free(p);
		return -1;
	}

	for (int i = 0; i < rows; i++) {
		(*matrix)[i] = &(p[i * cols]);
	}
	return 0;
}
//frees matrix
int freeMatrix(int*** matrix)
{
	free(&((*matrix)[0][0]));
	free(*matrix);
	return 0;
}
//matrix multiplication
void matrixMultiply(int** a, int** b, int rows, int cols, int*** c) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int val = 0;
			for (int k = 0; k < rows; k++) {
				val += a[i][k] * b[k][j];
			}
			(*c)[i][j] = val;
		}
	}
}
// prints matrix to terminal
void printMatrix(int** matrix, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("%d ", matrix[i][j]);
		}
		printf("\n");
	}
}
// prints matrix to file (fp)
void printMatrixFile(int** matrix, int size, FILE* fp) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			fprintf(fp, "%d ", matrix[i][j]);
		}
		fprintf(fp, "\n");
	}
}

int main(int argc, char* argv[]) {
	MPI_Comm cartComm;
	int dim[2], period[2], reorder;
	int coord[2], id;
	FILE *fp;
	int **A = NULL, **B = NULL, **C = NULL;
	int **localA = NULL, **localB = NULL, **localC = NULL;
	int **localARec = NULL, **localBRec = NULL;
	int rows = 0;
	int columns;
	int count = 0;
	int worldSize;
	int procDim;
	int blockDim;
	int left, right, up, down;
	int bCastData[4];

	// initialize MPI 
	MPI_Init(NULL, NULL);

	// world size
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
	
	
	int rank;
	// rank
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) {
		int n;
		char ch;

		// open file and determine the number of rows and columns
		fp = fopen("A.txt", "r");
		if (fp == NULL) {
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		while (fscanf(fp, "%d", &n) != EOF) {
			ch = fgetc(fp);
			if (ch == '\n') {
				rows = rows + 1;
			}
			count++;
		}
		columns = count / rows;
		printf("rows = %d, cols = %d\n", rows, columns);

		// check if matrix is square
		if (columns != rows) {
			printf("[ERROR] Matrix must be square!\n");
			MPI_Abort(MPI_COMM_WORLD, 2);
		}
		double sqroot = sqrt(worldSize);
		// check if number of processes is a perfect square
		if ((sqroot - floor(sqroot)) != 0) {
			printf("[ERROR] Number of processes must be a perfect square!\n");
			MPI_Abort(MPI_COMM_WORLD, 2);
		}
		int intRoot = (int)sqroot;
		// check if we can divide tasks between processes
		if (columns%intRoot != 0 || rows%intRoot != 0) {
			printf("[ERROR] Number of rows/columns not divisible by %d!\n", intRoot);
			MPI_Abort(MPI_COMM_WORLD, 3);
		}
		procDim = intRoot;
		blockDim = columns / intRoot;

		fseek(fp, 0, SEEK_SET);
		// allocates matrices to multiply
		if (allocMatrix(&A, rows, columns) != 0) {
			printf("[ERROR] Matrix alloc for A failed!\n");
			MPI_Abort(MPI_COMM_WORLD, 4);
		}
		if (allocMatrix(&B, rows, columns) != 0) {
			printf("[ERROR] Matrix alloc for B failed!\n");
			MPI_Abort(MPI_COMM_WORLD, 5);
		}

		// read first matrix from file
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				fscanf(fp, "%d", &n);
				A[i][j] = n;
			}
		}
		printf("A matrix:\n");
		printMatrix(A, rows);
		fclose(fp);

		// read second matrix from file
		fp = fopen("B.txt", "r");
		if (fp == NULL) {
			return 1;
		}
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				fscanf(fp, "%d", &n);
				B[i][j] = n;
			}
		}
		printf("B matrix:\n");
		printMatrix(B, rows);
		fclose(fp);
		// allocate the result matrix
		if (allocMatrix(&C, rows, columns) != 0) {
			printf("[ERROR] Matrix alloc for C failed!\n");
			MPI_Abort(MPI_COMM_WORLD, 6);
		}

		bCastData[0] = procDim;
		bCastData[1] = blockDim;
		bCastData[2] = rows;
		bCastData[3] = columns;
	}
	

	// create 2D cartesian grid of processes
	MPI_Bcast(&bCastData, 4, MPI_INT, 0, MPI_COMM_WORLD);
	procDim = bCastData[0];
	blockDim = bCastData[1];
	rows = bCastData[2];
	columns = bCastData[3];

	dim[0] = procDim; dim[1] = procDim;
	period[0] = 1; period[1] = 1;
	reorder = 1;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &cartComm);

	// allocate local blocks for matrices A and B
	allocMatrix(&localA, blockDim, blockDim);
	allocMatrix(&localB, blockDim, blockDim);

	// create datatype that describes subarrays of global array
	int globalSize[2] = { rows, columns };
	int localSize[2] = { blockDim, blockDim };
	int starts[2] = { 0,0 };
	MPI_Datatype type, subarrtype;
	MPI_Type_create_subarray(2, globalSize, localSize, starts, MPI_ORDER_C, MPI_INT, &type);
	MPI_Type_create_resized(type, 0, blockDim * sizeof(int), &subarrtype);
	MPI_Type_commit(&subarrtype);

	int *globalptrA = NULL;
	int *globalptrB = NULL;
	int *globalptrC = NULL;
	if (rank == 0) {
		globalptrA = &(A[0][0]);
		globalptrB = &(B[0][0]);
		globalptrC = &(C[0][0]);
	}

	// scatter array to al processors
	int* sendCounts = (int*)malloc(sizeof(int) * worldSize);
	int* displacements = (int*)malloc(sizeof(int) * worldSize);

	if (rank == 0) {
		for (int i = 0; i < worldSize; i++) {
			sendCounts[i] = 1;
		}
		int disp = 0;
		for (int i = 0; i < procDim; i++) {
			for (int j = 0; j < procDim; j++) {
				displacements[i * procDim + j] = disp;
				disp += 1;
			}
			disp += (blockDim - 1)* procDim;
		}
	}

	MPI_Scatterv(globalptrA, sendCounts, displacements, subarrtype, &(localA[0][0]),
		rows * columns / (worldSize), MPI_INT,
		0, MPI_COMM_WORLD);
	MPI_Scatterv(globalptrB, sendCounts, displacements, subarrtype, &(localB[0][0]),
		rows * columns / (worldSize), MPI_INT,
		0, MPI_COMM_WORLD);

	if (allocMatrix(&localC, blockDim, blockDim) != 0) {
		printf("[ERROR] Matrix alloc for localC in rank %d failed!\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 7);
	}

	// initial skew
	MPI_Cart_coords(cartComm, rank, 2, coord);
	MPI_Cart_shift(cartComm, 1, coord[0], &left, &right);
	MPI_Sendrecv_replace(&(localA[0][0]), blockDim * blockDim, MPI_INT, left, 1, right, 1, cartComm, MPI_STATUS_IGNORE);
	MPI_Cart_shift(cartComm, 0, coord[1], &up, &down);
	MPI_Sendrecv_replace(&(localB[0][0]), blockDim * blockDim, MPI_INT, up, 1, down, 1, cartComm, MPI_STATUS_IGNORE);

	// initialize matrix c with zeros
	for (int i = 0; i < blockDim; i++) {
		for (int j = 0; j < blockDim; j++) {
			localC[i][j] = 0;
		}
	}

	int** multiplyRes = NULL;
	if (allocMatrix(&multiplyRes, blockDim, blockDim) != 0) {
		printf("[ERROR] Matrix alloc for multiplyRes in rank %d failed!\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 8);
	}
	// matrix multiplication
	double start = MPI_Wtime();
	for (int k = 0; k < procDim; k++) {
		matrixMultiply(localA, localB, blockDim, blockDim, &multiplyRes);

		for (int i = 0; i < blockDim; i++) {
			for (int j = 0; j < blockDim; j++) {
				localC[i][j] += multiplyRes[i][j];
			}
		}
		// shift once A left and B up
		MPI_Cart_shift(cartComm, 1, 1, &left, &right);
		MPI_Cart_shift(cartComm, 0, 1, &up, &down);
		MPI_Sendrecv_replace(&(localA[0][0]), blockDim * blockDim, MPI_INT, left, 1, right, 1, cartComm, MPI_STATUS_IGNORE);
		MPI_Sendrecv_replace(&(localB[0][0]), blockDim * blockDim, MPI_INT, up, 1, down, 1, cartComm, MPI_STATUS_IGNORE);
	}
	
	// gathering results
	MPI_Gatherv(&(localC[0][0]), rows * columns / worldSize, MPI_INT,
		globalptrC, sendCounts, displacements, subarrtype,
		0, MPI_COMM_WORLD);

	double end = MPI_Wtime();

	// free the auxiliary matrices
	freeMatrix(&localC);
	freeMatrix(&multiplyRes);
	// open file to write results
	FILE * out = fopen("output.txt", "w");
	// print results
	if (rank == 0) {
		printf("C is:\n");
		printMatrix(C, rows);
		printf("\nExecution time is: %lf\n", end - start);
		printMatrixFile(C, rows, out);
		fprintf(out, "\nExecution time is: %lf\n", end - start);
	}
	// finalize the MPI
	MPI_Finalize();

	return 0;
}