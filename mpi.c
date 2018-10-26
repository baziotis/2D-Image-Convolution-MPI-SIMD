#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <stdint.h>
#include <assert.h>

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
	int sim_flag;
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
			// NOTE(maria): We have extra call on perfect squares.
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

	// NOTE(stefanos): We could do more exhausting
	// testing for the correctness of the input.
	input_data->input_file = calloc(strlen(argv[1]) + 1, sizeof(char));
	strcpy(input_data->input_file, argv[1]);
	if(my_rank == 0) {
		if(argc == 7) {
			input_data->width = atoi(argv[2]);
			input_data->height = atoi(argv[3]);
			input_data->bytes_per_pixel = atoi(argv[4]);
			input_data->times = atoi(argv[5]);
			// NOTE(maria): sim_flag refers to similarity check
			input_data->sim_flag = atoi(argv[6]);
			width_div = split_dimensions(input_data->width, input_data->height, comm_sz);
			if(!width_div) {
				fprintf(stderr, "[%s]: Could not split dimensions\n", argv[0]);
				success = 0;
			}
		} else {
			if(my_rank == 0)
				fprintf(stderr, "[%s]: Usage: %s [input_file] [width] [height] [bytes per pixel] [times] [sim_flag]\n", argv[0], argv[0]);
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
		MPI_Bcast(&(input_data->sim_flag), 1, MPI_INT, 0, MPI_COMM_WORLD);

		return width_div;
	}

	return 0;
}

///        PARALLEL I/O        ///

void Read_data(image_info_t *image_info, input_data_t *input_data, int start_row, int start_col, uint8_t *out) {
	// decouple struct data
	int cols = image_info->cols;
	int rows = image_info->rows;
	int bytes_per_pixel = image_info->bytes_per_pixel;

	char *input_file = input_data->input_file;
	int width = input_data->width;

	MPI_File in_file_handle;
	MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &in_file_handle);

	int read_pos;
	size_t size_of_one_line = cols * bytes_per_pixel;
	uint8_t *line_buffer = malloc(size_of_one_line * sizeof(uint8_t));
	for(int row = 0; row != rows; ++row) {
		int row_pos = (start_row + row) * width;
		read_pos = (row_pos + start_col) * bytes_per_pixel;
		MPI_File_seek(in_file_handle, read_pos, MPI_SEEK_SET);

		// read bytes
		MPI_File_read(in_file_handle, line_buffer, size_of_one_line, MPI_BYTE, MPI_STATUS_IGNORE);
		for(int i = 0; i != size_of_one_line; ++i)
			*out++ = (uint8_t) line_buffer[i];
	}

	free(line_buffer);

	MPI_File_close(&in_file_handle);
}

void Write_data(int my_rank, image_info_t *image_info, input_data_t *input_data, int start_row, int start_col, uint8_t *in) {
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
	size_t size_of_one_line = cols * bytes_per_pixel;
	uint8_t *line_buffer = malloc(size_of_one_line * sizeof(uint8_t));
	for(int row = 0; row != rows; ++row) {
		int row_pos = (start_row + row) * width;
		write_pos = (row_pos + start_col) * bytes_per_pixel;
		MPI_File_seek(out_file_handle, write_pos, MPI_SEEK_SET);

		for(int i = 0; i != size_of_one_line; ++i)
			line_buffer[i] = (uint8_t) *in++;

		// write bytes
		MPI_File_write(out_file_handle, line_buffer, size_of_one_line, MPI_BYTE, MPI_STATUS_IGNORE);
	}

	free(line_buffer);

	MPI_File_close(&out_file_handle);
}

///        COLOR MANIPULATION        ///

void Split_colors(image_info_t *image_info, uint8_t *in, uint8_t *out) {
	int rows = image_info->rows;
	int stride = image_info->cols;
	int bytes_per_pixel = image_info->bytes_per_pixel;

	uint8_t *reader;
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

void Recombine_colors(image_info_t *image_info, uint8_t *in, uint8_t *out) {
	int rows = image_info->rows;
	int stride = image_info->cols;
	int bytes_per_pixel = image_info->bytes_per_pixel;

	uint8_t *writer;
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

// NOTE(maria): If at least 1 pixel is diff
// we do not need to check again
int Check_similarity(uint8_t *cache_in, uint8_t *cache_out, int pos){
    if (cache_in[pos] != cache_out[pos])
        return 1;
    return 0;
}

///        CONVOLUTION       ///

int fill_pixels(int curr_row, int curr_col, int width, uint8_t *start_data, uint8_t *cache_out, float *conv_matrix, int check, int check_similarity) {
	float pixel = 0;
	int k = 0;
	// Gather the 8 surrounding pixels for each source pixel.
	for(int i = curr_row - 1; i <= curr_row + 1; ++i)
		for(int j = curr_col - 1; j <= curr_col + 1; ++j)
			pixel += start_data[i * width + j] * conv_matrix[k++];

	cache_out[curr_row * width + curr_col] = pixel;

 	if ((check_similarity) && (!check))
    	check = Check_similarity(start_data,cache_out,(curr_row * width + curr_col));

	return check;
}

int compute(uint8_t *cache_in, uint8_t *cache_out, int start_row, int end_row, int start_col, int end_col, int width, float *convolution_matrix, long int avail_threads, int check_similarity) {

    int check = 0;
	int row, col;
	for(row = start_row; row <= end_row; ++row)
		for(col = start_col; col <= end_col; ++col)
			  //NOTE(maria): check refers to whether img is changed or not
		      check = fill_pixels(row, col, width, cache_in, cache_out, convolution_matrix, check, check_similarity);
    return check;
}

void normalize_kernel(float *conv_matrix) {
	float sum = 0.0;
	for(int i = 0; i < 9; ++i)
		sum += conv_matrix[i];

	for(int i = 0; i < 9; ++i)
		conv_matrix[i] /= sum;
}

int main(int argc, char **argv) {

	int	comm_sz;	// number of processes
	int my_rank; 	// my process rank

	double local_elapsed, elapsed;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	// gaussian blur
	float convolution_matrix[9] = { 1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0 };
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

    int bytes_per_pixel = image_info.bytes_per_pixel;
	int cols = image_info.cols;
	int rows = image_info.rows;
	int times = input_data.times;
	int check_similarity = input_data.sim_flag;

	// Track where each process's rectangle is in the whole image.
	start_row = (my_rank / width_div) * image_info.rows;
	start_col = (my_rank % width_div) * image_info.cols;

	// NOTE(stefanos): 2 padding lines, one above and one below the valid ones.
	// Also, 2 padding pixels for each valid line, one left, one right.
	int per_process_bytes = image_info.bytes_per_pixel * (image_info.rows + 2) * (image_info.cols + 2);
	uint8_t *src = calloc(per_process_bytes, sizeof(uint8_t));
	uint8_t *dst = calloc(per_process_bytes, sizeof(uint8_t));

	uint8_t *buffer = malloc(image_info.rows * image_info.cols * image_info.bytes_per_pixel * sizeof(uint8_t));

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

	// Type to send one whole padding column
	MPI_Type_vector(bytes_per_pixel * (rows+2), 1, cols+2, MPI_BYTE, &col_type);
	MPI_Type_commit(&col_type);
	// Type to send bytes_per_pixel rows, each of whome is 1 color's bytes worth (including the padding) apart.
	MPI_Type_vector(bytes_per_pixel, cols, (rows+2)*(cols+2), MPI_BYTE, &row_type);
	MPI_Type_commit(&row_type);

	/// Compute neighbors. ///

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

    // 0: top   1: bottom
    // 2: left  3: right
    MPI_Request send_req[4], recv_req[4];

	MPI_Barrier(MPI_COMM_WORLD);
	local_elapsed = MPI_Wtime();

	for(int t = 0; t != times; ++t) {

		// top
		MPI_Isend(src + (cols+2) + 1, 1, row_type, top, 0, MPI_COMM_WORLD, &send_req[0]);
		MPI_Irecv(src + 1, 1, row_type, top, 0, MPI_COMM_WORLD, &recv_req[0]);

		// bottom
		MPI_Isend(src + rows*(cols+2) + 1, 1, row_type, bottom, 0, MPI_COMM_WORLD, &send_req[1]);
		MPI_Irecv(src + (rows+1)*(cols+2) + 1, 1, row_type, bottom, 0, MPI_COMM_WORLD, &recv_req[1]);

		// left
		MPI_Isend(src + 1, 1, col_type, left, 0, MPI_COMM_WORLD, &send_req[2]);
		MPI_Irecv(src , 1, col_type, left, 0, MPI_COMM_WORLD, &recv_req[2]);

		// right
		MPI_Isend(src + (cols+2) - 2, 1, col_type, right, 0, MPI_COMM_WORLD, &send_req[3]);
		MPI_Irecv(src + (cols+2) - 1, 1, col_type, right, 0, MPI_COMM_WORLD, &recv_req[3]);

		// compute inner data
        int local_sim_flag;
		for(int color = 0; color != bytes_per_pixel; ++color) {
			// NOTE(maria): We check similarity only in inner data conv
		    local_sim_flag = compute(src, dst, color * (rows+2) + 1, (color+1) * (rows+2) - 2,
				1, cols, cols + 2, convolution_matrix, 0, check_similarity);
		}

		MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
		MPI_Wait(&recv_req[1], MPI_STATUS_IGNORE);
		MPI_Wait(&recv_req[2], MPI_STATUS_IGNORE);
		MPI_Wait(&recv_req[3], MPI_STATUS_IGNORE);

		/// Compute outer data ///
		// NOTE(maria): Last parameter is initilized to 0
		// in order to skip similarity_check
		if(top != MPI_PROC_NULL) {
			for(int color = 0; color != bytes_per_pixel; ++color) {
				compute(src, dst, color * (rows+2) + 1, color * (rows+2) + 1,
					1, cols, cols + 2, convolution_matrix, 0, 0);
			}
		}

		if(bottom != MPI_PROC_NULL) {
			for(int color = 0; color != bytes_per_pixel; ++color) {
				compute(src, dst, (color+1) * (rows+2) - 2, (color+1) * (rows+2) - 2,
					1, cols, cols + 2, convolution_matrix, 0, 0);
			}
		}

		if(left != MPI_PROC_NULL) {
			for(int color = 0; color != bytes_per_pixel; ++color) {
				compute(src, dst, color * (rows+2) + 1, (color+1) * (rows+2) - 2,
					1, 1, cols + 2, convolution_matrix, 0, 0);
			}
		}

		if(right != MPI_PROC_NULL) {
			for(int color = 0; color != bytes_per_pixel; ++color) {
				compute(src, dst, color * (rows+2) + 1, (color+1) * (rows+2) - 2,
					cols, cols, cols + 2, convolution_matrix, 0, 0);
			}
		}

		MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
		MPI_Wait(&send_req[1], MPI_STATUS_IGNORE);
		MPI_Wait(&send_req[2], MPI_STATUS_IGNORE);
		MPI_Wait(&send_req[3], MPI_STATUS_IGNORE);

		// Check for similarity
		// between src and dst image
		if(check_similarity){
            int global_sum;
            MPI_Allreduce(&local_sim_flag, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
			//if sum == 0 none part of img
			//is changed after convolution
			if(global_sum == 0)
                break;
		}

        // Swap arrays
		uint8_t *temp = src;
		src = dst;
		dst = temp;
	}

	local_elapsed = MPI_Wtime() - local_elapsed;
	MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(my_rank == 0) {
		fprintf(stderr, "Time for computation: %.15lf seconds\n", elapsed);
	}

	/// Write Data ///
	MPI_Barrier(MPI_COMM_WORLD);
	local_elapsed = MPI_Wtime();

	Recombine_colors(&image_info, src, buffer);
	Write_data(my_rank, &image_info, &input_data, start_row, start_col, buffer);

	local_elapsed = MPI_Wtime() - local_elapsed;
	MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(my_rank == 0)
		fprintf(stderr, "Read Data: %.15lf seconds\n", elapsed);

	free(src);
	free(dst);
	free(input_data.input_file);

	MPI_Finalize();
	return 0;
}
