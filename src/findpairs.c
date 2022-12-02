#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>

char * prefix = ".";	/* default output prefix */

#define noDuplicates 1

/* return the squared Euclidean distance between ci and cj */

inline float distance(float * ci, float * cj, int cols, int k) {
	float diff = 0;
	for (int m=0; m<cols; m++) {
		if (m != k) {
			float d = ci[m] - cj[m];
			diff += d * d;
		}
	}
	return diff;
}

/* return the categorical distance, ie. number of attributes with differing values */

inline float distanceCategorical(float * ci, float * cj, int cols, int k) {
	float diff = 0;
	for (int m=0; m<cols; m++) {
		if (m != k && ci[m] != cj[m])
			diff += 1;
	}
	return diff;
}

int categorical = 0;
int attribute = -1;

void error(char * message, char * option) {
	fprintf(stderr, message, option);
	exit(1);
}

char * getArg(char ** argv, int argc, int * arg, char * reason) {
	if (*arg < argc)
		return argv[(*arg)++];
	error("Insufficient arguments for %s\n", reason);
}

int main(int argc, char ** argv) {
	/* handle arguments */
	char * file = argv[1];
	float threshold = atof(argv[2]);
	int limit = atoi(argv[3]);
	int arg = 4;
	while (arg < argc) {
		char * opt = getArg(argv, argc, &arg, "option");
		if (!strcmp(opt, "--prefix")) 
			prefix = getArg(argv, argc, &arg, "prefix");
		else if (!strcmp(opt, "--categorical"))
			categorical = 1;
		else if (!strcmp(opt, "--attribute"))
			attribute = atoi(getArg(argv, argc, &arg, "attribute"));
		else
			error("Unknown option: %s\n", opt);
	}

	printf("threshold=%f\n", threshold);
	printf("limit=%d\n", limit);
	if (!categorical)
		threshold = threshold * threshold;

	/* read data */

	FILE * f = fopen(file, "r");
	if (!f) {
		fprintf(stderr, "Cannot open %s\n", file);
		exit(1);
	}

	int rows, cols;
	assert(fscanf(f, "%d %d\n", &rows, &cols)==2);
	printf("rows=%d, cols=%d\n", rows, cols);

	assert(attribute >= -1 && attribute<cols);

	printf("Reading data..."); fflush(stdout);
	float * data = (float *) malloc(rows * cols * sizeof(float));
	float * p = data;
	for (int i=0; i<rows; i++)
		for (int j=0; j<cols; j++) {
			assert(fscanf(f, "%f", p)==1);
			++p;
		}
	fclose(f);
	printf("done\n");

#if 0
	/* check read data */
	f = fopen("check.csv", "w");
	for (int i=0; i<rows; i++) {
		fprintf(f, "%d", i);
		for (int j=0; j<cols; j++)
			fprintf(f, " %f", data[i*cols + j]);
		fprintf(f, "\n");
	}
	fclose(f);
#endif

	int * index = malloc(rows * sizeof(int));

	for (int k=0; k<cols; k++) {
		if (attribute >= 0 && attribute != k)
			continue;
		time_t start = time(NULL);
		char fileName[1000];
		snprintf(fileName, 1000, "%s/p_%d.csv", prefix, k);
		printf("Attribute %d -> %s\n", k, fileName);
		f = fopen(fileName, "w");
		if (!f) {
			fprintf(stderr, "Cannot open output file %s\n", fileName);
			exit(1);
		}

		for (int i=0; i<rows; i++)
			index[i] = 1;

		int count = 0;
		for (int i=0; i<rows; i++) {
			if (!index[i]) continue;
			float * ci = data + i * cols;
			for (int j=0; j < i; j++) {
				if (!index[j]) continue;
				float * cj = data + j * cols;
				float diff = categorical ? 
					distanceCategorical(ci, cj, cols, k) : distance(ci, cj, cols, k);
				if (diff <= threshold) {
					fprintf(f, "%d,%d,%f\n", i, j, diff);
#if DEBUG
					for (int m=0; m<cols; m++) fprintf(f, "%f ", ci[m]);
					fprintf(f, "\n");
					for (int m=0; m<cols; m++) fprintf(f, "%f ", cj[m]);
					fprintf(f, "\n");
#endif
					++count;
					if (count % 100 == 0) {
						time_t diff = time(NULL) - start;
						float rate = 0;
						if (diff>0)
							rate = count / (float)diff;
						int progress = (int)(100*pow(i / (float) rows, 2));
						printf("  %d (%.2f per second, progress %d%%)\r", count, rate, progress); fflush(stdout);
					}
					if (count == limit) {
						printf("\n  Limit reached with progress=%d/%d", i, rows);
						j = i = rows;
					}
					if (noDuplicates) {
						index[i] = index[j] = 0;
						j = i;
					}
				}
			}
		}
		time_t diff = time(NULL) - start;
		printf("\n  Generated %d pairs in %ld seconds\n", count, diff);
		fclose(f);
	}
}

