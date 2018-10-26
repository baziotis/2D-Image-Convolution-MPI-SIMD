#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <stdint.h>
#include <assert.h>
#include <immintrin.h>

#define KERNEL_SIZE 3

typedef struct image_info {
	int cols;
	int rows;
	int bytes_per_pixel;
} image_info_t;

typedef struct input_data {
	int width;
	int height;
	int bytes_per_pixel;
	int times;
	char *input_file;
} input_data_t;


///        DIMENSION DIVISION AND USAGE        ///

void split_helper(int width, int height, int ps, int width_div, int *pbest_div, int *pper_min) {
	int best_div, per_min;
	int height_div;

	best_div = *pbest_div;
	per_min = *pper_min;

	if(width % width_div == 0) {
		height_div = ps / width_div;
		if(height % height_div == 0) {
			int curr_per = width / width_div + height / height_div;
			if(curr_per < per_min) {
				per_min = curr_per;
				best_div = width_div;
			}
		}
	}

	*pbest_div = best_div;
	*pper_min = per_min;
}

// Divide image in 'ps' equal rectangles so that the perimeter
// of each rectangle is minimized (in order to minimize
// the exchange of data between processes).
// This procedure is serial, it's supposed to be called from process 0.
int split_dimensions(int width, int height, int ps) {
	int width_div;
	int best_div, per_min;
	int inc;

	best_div = 0;
	per_min = height + width + 1;

	inc = 1;
	if(width % 2)
		inc = 2;
	for(width_div = 1; width_div*width_div <= ps; width_div += inc) {
		if(!(ps % width_div)) {
			// NOTE(stefanos): We have extra call on perfect squares.
			split_helper(width, height, ps, width_div, &best_div, &per_min);
			split_helper(width, height, ps, ps / width_div, &best_div, &per_min);
		}
	}

	return best_div;
}

// Check and broadcast command line arguments
// On success, return width divisor
// On failure, return 0
int Get_input(int my_rank, int comm_sz, int argc, char **argv, input_data_t *input_data) {
	int success, width_div;
	success = 1;

	input_data->input_file = calloc(strlen(argv[1]) + 1, sizeof(char));
	strcpy(input_data->input_file, argv[1]);
	if(my_rank == 0) {
		if(argc == 6) {
			input_data->width = atoi(argv[2]);
			input_data->height = atoi(argv[3]);
			input_data->bytes_per_pixel = atoi(argv[4]);
			input_data->times = atoi(argv[5]);

			width_div = split_dimensions(input_data->width, input_data->height, comm_sz);
			if(!width_div) {
				fprintf(stderr, "[%s]: Could not split dimensions\n", argv[0]);
				success = 0;
			}
		} else {
			if(my_rank == 0)
				fprintf(stderr, "[%s]: Usage: %s [input_file] [width] [height] [bytes per pixel] [times]\n", argv[0], argv[0]);
			success = 0;
		}
	}

	MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(success) {
		// Broadcast width divisor so that every process can compute its
		// rows and cols.
		MPI_Bcast(&width_div, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&(input_data->width), 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&(input_data->height), 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&(input_data->bytes_per_pixel), 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&(input_data->times), 1, MPI_INT, 0, MPI_COMM_WORLD);

		return width_div;
	}

	return 0;
}


///        PARALLEL I/O        ///

void Read_data(image_info_t *image_info, input_data_t *input_data, int start_row, int start_col, float *out) {

	int cols = image_info->cols;
	int rows = image_info->rows;
	int bytes_per_pixel = image_info->bytes_per_pixel;

	char *input_file = input_data->input_file;
	int width = input_data->width;

	MPI_File in_file_handle;
	MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &in_file_handle);

	int read_pos;
	float *temp = malloc(cols * bytes_per_pixel * sizeof(float));
	for(int row = 0; row != rows; ++row) {
		read_pos = ((start_row + row) * width + start_col) * bytes_per_pixel;
		MPI_File_seek(in_file_handle, read_pos, MPI_SEEK_SET);

		// read in bytes
		MPI_File_read(in_file_handle, temp, bytes_per_pixel * cols, MPI_BYTE, MPI_STATUS_IGNORE);
		// copy and convert to floats
		for(int i = 0; i != bytes_per_pixel * cols; ++i)
			*out++ = (float) temp[i];
	}

	free(temp);

	MPI_File_close(&in_file_handle);
}

void Write_data(int my_rank, image_info_t *image_info, input_data_t *input_data, int start_row, int start_col, float *in) {
	// decouple struct data
	int cols = image_info->cols;
	int rows = image_info->rows;
	int bytes_per_pixel = image_info->bytes_per_pixel;
	int width = input_data->width;

	char out_image[64];
	strcpy(out_image, "test_out.raw");

	MPI_File out_file_handle;
	MPI_File_open(MPI_COMM_WORLD, out_image, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out_file_handle);

	int write_pos;
	float *temp = malloc(cols * bytes_per_pixel * sizeof(float));
	for(int row = 0; row != rows; ++row) {
		write_pos = ((start_row + row) * width + start_col) * bytes_per_pixel;
		MPI_File_seek(out_file_handle, write_pos, MPI_SEEK_SET);
		// copy and convert to floats
		for(int i = 0; i != bytes_per_pixel * cols; ++i)
			temp[i] = (float) *in++;
		// read in bytes
		MPI_File_write(out_file_handle, temp, bytes_per_pixel * cols, MPI_BYTE, MPI_STATUS_IGNORE);
	}

	free(temp);

	MPI_File_close(&out_file_handle);
}


///        COLOR MANIPULATION        ///


// Split colors so that bytes of the same color are packed together (So, first the bytes
// of red, then green and so on...)
void Split_colors(image_info_t *image_info, float *in, float *out) {
	int rows = image_info->rows;
	int stride = image_info->cols;
	int bytes_per_pixel = image_info->bytes_per_pixel;

	float *reader;
	// skip the first padding line
	out += stride + 2;
	// For every color
	for(int color = 0; color != bytes_per_pixel; ++color) {
		reader = in + color;  // start at the ith (1,2,3,4) byte of the first pixel
		// for every row
		for(int row = 0; row != rows; ++row) {
			++out;  // skip one padding pixel
			// NOTE(stefanos): For each color, each of its bytes is bytes_per_pixel
			// apart from the next.
			for(int col = 0; col != stride; ++col) {
				*out++ = *reader;
				reader += bytes_per_pixel;
			}
			++out;  // skip one padding pixel
		}
		// skip 2 intermediate padding lines
		out += 2 * (stride + 2);
	}
}

// Revert color packing to the original structure (i.e. RGB RGB RGB ...)
void Recombine_colors(image_info_t *image_info, float *in, float *out) {
	int rows = image_info->rows;
	int stride = image_info->cols;
	int bytes_per_pixel = image_info->bytes_per_pixel;

	float *writer;
	// skip the first padding line
	in += stride + 2;
	for(int color = 0; color != bytes_per_pixel; ++color) {
		writer = out + color;
		for(int row = 0; row != rows; ++row) {
			++in;  // skip one padding pixel
			// NOTE(stefanos): For each color, each of its bytes is bytes_per_pixel
			// apart from the next.
			for(int col = 0; col != stride; ++col) {
				*writer = *in++;
				writer += bytes_per_pixel;
			}
			++in;  // skip one padding pixel
		}
		// skip 2 intermediate padding lines
		in += 2 * (stride + 2);
	}
}

///        CONVOLUTION       ///

void fill_pixels(int curr_row, int curr_col, int width, float *start_data, float *cache_out, float *conv_matrix) {
	float pixel = 0;
	int k = 0;
	// Gather the 8 surrounding pixels for each source pixel.
	for(int i = curr_row - 1; i <= curr_row + 1; ++i)
		for(int j = curr_col - 1; j <= curr_col + 1; ++j)
			pixel += start_data[i * width + j] * conv_matrix[k++];

	cache_out[curr_row * width + curr_col] = pixel;
}

void compute(float *cache_in, float *cache_out, int start_row, int end_row, int start_col, int end_col, int width, float *convolution_matrix, long int avail_threads) {

	int row, col;

	for(row = start_row; row <= end_row; ++row)
		for(col = start_col; col <= end_col; ++col)
			fill_pixels(row, col, width, cache_in, cache_out, convolution_matrix);
}


// 1D convolution.
// Assume that aligned_out is aligned to 32 byte boundary.
void avx_convolve(float *in, float *aligned_out, int length, float kernel[KERNEL_SIZE]) {

// Get aligned (to 32) local variables depending on the compiler.
#ifdef __GNUC__

	__m256 kernel_vec[KERNEL_SIZE] __attribute__((aligned(32)));
	__m256 data_block __attribute__((aligned(32)));

	__m256 prod __attribute__ ((aligned(32)));
	__m256 acc __attribute__ ((aligned(32)));

#endif

#ifdef _MSC_VER
	__declspec(align(32)) __m256 kernel_vec[KERNEL_SIZE];
	__declspec(align(32)) __m256 data_block;

	__declspec(align(32)) __m256 prod;
	__declspec(align(32)) __m256 acc;
#endif

	int i, k;

	// Repeat each kernel value in a 4-wide register
	for(i = 0; i < KERNEL_SIZE; ++i) {
		kernel_vec[i] = _mm256_set1_ps(kernel[i]);
	}

	for(i = 0; i < length - 8; i+=8) {

		// Zero accumulator
		acc = _mm256_setzero_ps();

		// NOTE(stefanos): With optimizations,
		// this loop is unrolled by the compiler
		for(k = -1; k <= 1; ++k) {
			// Load 8-float data block (unaligned access)
			data_block = _mm256_loadu_ps(in + i + k);
			prod = _mm256_mul_ps(kernel_vec[k + 1], data_block);

			// Accumulate the 8 parallel values
			acc = _mm256_add_ps(acc, prod);
		}

		// Stores are aligned because aligned_out is
		// aligned_out is aligned to a 32-byte boundary
		// and we go +8 every time.
		_mm256_store_ps(aligned_out + i, acc);
	}

	// Scalar computation for the rest < 8 pixels.
	while(i != length) {
		aligned_out[i] = 0.0;
		for(k = -1; k <= 1; ++k)
			aligned_out[i] += in[i + k] * kernel[k+1];
		++i;
	}
}

// 2D convolution.
// Assume that each line in 'lines' is aligned to 32 byte boundary.
void simd_compute(float *cache_in, float *cache_out, int start_row, int end_row, int start_col, int end_col, int width, float *convolution_matrix, float *lines[3]) {

// Get aligned (to 32) local variables depending on the compiler.
#ifdef __GNUC__

	__m256 sum1 __attribute__((aligned(32)));
	__m256 sum2 __attribute__((aligned(32)));

#endif

#ifdef _MSC_VER

	__declspec(align(32)) __m256 sum1;
	__declspec(align(32)) __m256 sum2;

#endif


	for(int row = start_row; row <= end_row; ++row) {
		// Compute 3 1D convolutions.
		avx_convolve(cache_in + (row - 1) * width + start_col, lines[0], width - 2, convolution_matrix);
		avx_convolve(cache_in + row * width + start_col, lines[1], width - 2, convolution_matrix + 3);
		avx_convolve(cache_in + (row + 1) * width + start_col, lines[2], width - 2, convolution_matrix + 6);

		int i, k;
		for(i = start_col, k = 0; i <= end_col - 8; k+=8, i+=8) {
		// NOTE(stefanos): Loads here can be aligned because we go in groups of 8
		// and start from the 0th element and lines have been allocated to a 32 byte boundary.
			sum1 = _mm256_add_ps(_mm256_load_ps(&lines[0][k]), _mm256_load_ps(&lines[1][k]));
			sum2 = _mm256_add_ps(sum1, _mm256_load_ps(&lines[2][k]));
			_mm256_storeu_ps(&cache_out[row * width + i], sum2);
		}

		// Handle what has remained in scalar.
		while(i <= end_col) {
			cache_out[row * width + i] = lines[0][k] + lines[1][k] + lines[2][k];
			++i;
			++k;
		}
	}
}

void normalize_kernel(float *conv_matrix) {
	float sum = 0.0f;
	for(int i = 0; i < 9; ++i)
		sum += conv_matrix[i];

	for(int i = 0; i < 9; ++i)
		conv_matrix[i] /= sum;
}


int main(int argc, char **argv) {

	int		comm_sz;	// number of processes
	int 	my_rank; 	// my process rank

	double local_elapsed, elapsed;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	// gaussian blur
	float convolution_matrix[9] =
	{
		1.0, 2.0, 1.0,
		2.0, 4.0, 2.0,
		1.0, 2.0, 1.0
	};

	normalize_kernel(convolution_matrix);

	int width_div;
	input_data_t input_data;

	width_div = Get_input(my_rank, comm_sz, argc, argv, &input_data);

	if(!width_div) {
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	image_info_t image_info;
	int start_row, start_col;

	image_info.rows = input_data.height / (comm_sz / width_div);
	image_info.cols = input_data.width / width_div;
	image_info.bytes_per_pixel = input_data.bytes_per_pixel;

	// Track where each process's rectangle is in the whole image.
	start_row = (my_rank / width_div) * image_info.rows;
	start_col = (my_rank % width_div) * image_info.cols;

	// NOTE(stefanos): 2 padding lines, one above and one below the valid ones.
	// Also, 2 padding pixels for each valid line, one left, one right.
	int per_process_bytes = image_info.bytes_per_pixel * (image_info.rows + 2) * (image_info.cols + 2);
	float *src = calloc(per_process_bytes, sizeof(float));
	float *dst = calloc(per_process_bytes, sizeof(float));

	float *buffer = malloc(image_info.rows * image_info.cols * image_info.bytes_per_pixel * sizeof(float));

	/// Read Data ///
	MPI_Barrier(MPI_COMM_WORLD);
	local_elapsed = MPI_Wtime();

	Read_data(&image_info, &input_data, start_row, start_col, buffer);
	Split_colors(&image_info, buffer, src);

	local_elapsed = MPI_Wtime() - local_elapsed;
	MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(my_rank == 0) {
		fprintf(stderr, "Read Data: %.15lf seconds\n", elapsed);
	}


    MPI_Datatype col_type;
    MPI_Datatype row_type;

	int bytes_per_pixel = image_info.bytes_per_pixel;
	int cols = image_info.cols;
	int rows = image_info.rows;
	int times = input_data.times;

    MPI_Request top_req_send;
    MPI_Request bottom_req_send;
    MPI_Request left_req_send;
    MPI_Request right_req_send;
    MPI_Request top_req_recv;
    MPI_Request bottom_req_recv;
    MPI_Request left_req_recv;
    MPI_Request right_req_recv;

	// Type to send one whole column.
	MPI_Type_vector(bytes_per_pixel * (rows+2), 1, cols+2, MPI_FLOAT, &col_type);
	MPI_Type_commit(&col_type);
	// Type to send bytes_per_pixel rows, each of whome is 1 color's bytes worth (including the padding) apart.
	MPI_Type_vector(bytes_per_pixel, cols, (rows+2)*(cols+2), MPI_FLOAT, &row_type);
	MPI_Type_commit(&row_type);

	// Compute neighbors.

	// Initialization to null process, i.e. no neighbor.
	int top = MPI_PROC_NULL;
	int bottom = MPI_PROC_NULL;
	int left = MPI_PROC_NULL;
	int right = MPI_PROC_NULL;

	if(start_row != 0)
		top = my_rank - width_div;
	if(start_row + image_info.rows != input_data.height)
		bottom = my_rank + width_div;
	if(start_col != 0)
		left = my_rank - 1;
	if(start_col + image_info.cols != input_data.width)
		right = my_rank + 1;

	float *lines[3];

// Aligned heap allocation depending on the compiler. That is actually not dependent on the
// compiler but on the OS (POSIX or Windows).
#	ifdef __GNUC__
	assert(posix_memalign((void **) &lines[0], 32, image_info.cols * sizeof(float)) == 0);
	assert(posix_memalign((void **) &lines[1], 32, image_info.cols * sizeof(float)) == 0);
	assert(posix_memalign((void **) &lines[2], 32, image_info.cols * sizeof(float)) == 0);
#	elif defined(_MSC_VER)
	assert((lines[0] = _aligned_malloc(image_info.cols * sizeof(float), 32)) != NULL);
	assert((lines[1] = _aligned_malloc(image_info.cols * sizeof(float), 32)) != NULL);
	assert((lines[2] = _aligned_malloc(image_info.cols * sizeof(float), 32)) != NULL);
#	endif

	MPI_Barrier(MPI_COMM_WORLD);
	local_elapsed = MPI_Wtime();

	for(int t = 0; t != times; ++t) {
		// top
		MPI_Isend(src + (cols+2) + 1, 1, row_type, top, 0, MPI_COMM_WORLD, &top_req_send);
		MPI_Irecv(src + 1, 1, row_type, top, 0, MPI_COMM_WORLD, &top_req_recv);

		// bottom
		MPI_Isend(src + rows*(cols+2) + 1, 1, row_type, bottom, 0, MPI_COMM_WORLD, &bottom_req_send);
		MPI_Irecv(src + (rows+1)*(cols+2) + 1, 1, row_type, bottom, 0, MPI_COMM_WORLD, &bottom_req_recv);

		// left
		MPI_Isend(src + 1, 1, col_type, left, 0, MPI_COMM_WORLD, &left_req_send);
		MPI_Irecv(src , 1, col_type, left, 0, MPI_COMM_WORLD, &left_req_recv);

		// right
		MPI_Isend(src + (cols+2) - 2, 1, col_type, right, 0, MPI_COMM_WORLD, &right_req_send);
		MPI_Irecv(src + (cols+2) - 1, 1, col_type, right, 0, MPI_COMM_WORLD, &right_req_recv);

		// compute inner data
		for(int color = 0; color != bytes_per_pixel; ++color) {
			simd_compute(src, dst, color * (rows+2) + 1, (color+1) * (rows+2) - 2,
				1, cols, cols + 2, convolution_matrix, lines);
		}

		MPI_Wait(&top_req_recv, MPI_STATUS_IGNORE);
		MPI_Wait(&bottom_req_recv, MPI_STATUS_IGNORE);
		MPI_Wait(&left_req_recv, MPI_STATUS_IGNORE);
		MPI_Wait(&right_req_recv, MPI_STATUS_IGNORE);

		// Compute outer data
		if(top != MPI_PROC_NULL) {
			for(int color = 0; color != bytes_per_pixel; ++color) {
				compute(src, dst, color * (rows+2) + 1, color * (rows+2) + 1,
					1, cols, cols + 2, convolution_matrix, 0);
			}
		}

		if(bottom != MPI_PROC_NULL) {
			for(int color = 0; color != bytes_per_pixel; ++color) {
				compute(src, dst, (color+1) * (rows+2) - 2, (color+1) * (rows+2) - 2,
					1, cols, cols + 2, convolution_matrix, 0);
			}
		}

		if(left != MPI_PROC_NULL) {
			for(int color = 0; color != bytes_per_pixel; ++color) {
				compute(src, dst, color * (rows+2) + 1, (color+1) * (rows+2) - 2,
					1, 1, cols + 2, convolution_matrix, 0);
			}
		}

		if(right != MPI_PROC_NULL) {
			for(int color = 0; color != bytes_per_pixel; ++color) {
				compute(src, dst, color * (rows+2) + 1, (color+1) * (rows+2) - 2,
					cols, cols, cols + 2, convolution_matrix, 0);
			}
		}

		MPI_Wait(&top_req_send, MPI_STATUS_IGNORE);
		MPI_Wait(&bottom_req_send, MPI_STATUS_IGNORE);
		MPI_Wait(&left_req_send, MPI_STATUS_IGNORE);
		MPI_Wait(&right_req_send, MPI_STATUS_IGNORE);

		float *temp = src;
		src = dst;
		dst = temp;
	}

	local_elapsed = MPI_Wtime() - local_elapsed;
	MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(my_rank == 0) {
		fprintf(stderr, "Time for computation: %.15lf seconds\n", elapsed);
	}

// In Windows, memory freeing is different depending on how the memory was allocated.
#	ifdef __GNUC__
	free(lines[0]);
	free(lines[1]);
	free(lines[2]);
#	elif defined(_MSC_VER)
	_aligned_free(lines[0]);
	_aligned_free(lines[1]);
	_aligned_free(lines[2]);
#	endif

	/// Write Data ///
	MPI_Barrier(MPI_COMM_WORLD);
	local_elapsed = MPI_Wtime();

	Recombine_colors(&image_info, src, buffer);
	Write_data(my_rank, &image_info, &input_data, start_row, start_col, buffer);

	local_elapsed = MPI_Wtime() - local_elapsed;
	MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(my_rank == 0) {
		fprintf(stderr, "Read Data: %.15lf seconds\n", elapsed);
	}

	free(src);
	free(dst);
	free(input_data.input_file);

	MPI_Finalize();
	return 0;
}
