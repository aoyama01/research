/*
   Fast algorithm
            for CDMA2

         Ken KIYONO
         5 May 2019
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <time.h>

#define SWAP(a,b) {temp = (a); (a) = (b); (b) = temp;}

long read_data(char filename[], int skip, int selcol1,int selcol2);
void help(void);
void error(char error_text[]);
double *vector(long nl, long nh);
int *ivector(long nl, long nh);
long *lvector(long nl, long nh);
void free_vector(double *v, long nl, long nh);
void free_ivector(int *v, long nl, long nh);
void free_lvector(long *v, long nl, long nh);

double est_f2(long n,long scale);
void cmat(double k);
double c1;


/* Global variables */
char *pname; /* this program name */
double *data1,*y01; /* input data and integrated data */
double *data2,*y02;
double f2_11,f2_22;
long lag; 

long rslen;	/* length of rs[] */
long *rs;
long nr;	/* number of box sizes */
long i_refresh;

main(int argc, char **argv)
{
	double *F2_12,*F_11,*F_22;
	long n; /* data length*/
	long i;
	long s, c1,c2; /* s: skiped lines, c: selected column */
	long minbox, maxbox;
	clock_t start, end;

	/* default values for selected data column */
	s = 0;
	c1 = 1;
	c2 = 2;
	i_refresh = 50000;
	minbox = 5;
	maxbox = 0;
	lag=0;

	/* options */
	pname = argv[0];
	for (i = 1; i < argc && *argv[i] == '-'; i++) {
		switch(argv[i][1]) {
		case 's':
			s = atoi(argv[++i]);
			break;
		case 'L':
			lag = atol(argv[++i]);
			break;
		case 'c':
			c1 = atoi(argv[++i]);
			c2 = atoi(argv[++i]);
			break;
		case 'r':
			i_refresh = atol(argv[++i]);
			break;
		case 'l':	/* set minbox (the minimum box size) */
			minbox = atol(argv[++i]); break;
		case 'u':	/* set maxbox (the maximum box size) */
			maxbox = atol(argv[++i]); break;
	   case 'h':	/* print usage information and quit */
           default:
	      help();
	      exit(1);
	      }
	}

	/* read data file */
	n = read_data(argv[argc-1], s, c1,c2);
	fprintf(stderr, "Total data length = %ld\n", n); 
	fprintf(stderr, "***** DMA0 *****\n"); 

	/* Set minimum and maximum box sizes. */
	minbox = (long)(minbox/2)*2+1; /* minbox should be an odd number */
	if(minbox < 5) minbox = 5;
	if(maxbox > n/2 || minbox > maxbox) maxbox = (long)n/2;
	
	/* "rscale" is based on that of "dfa.c". */
	/* Note that scales are odd numbers in CDMA. */
	nr = rscale(minbox, maxbox, pow(2.0, 1.0/8.0));

	/* For measurement of processing time */
	start = clock();
	fprintf(stderr,"starting time:%d\n", start);
	
	/* define fluctuation function */
	F_11 = vector(1,nr);	
	F_22 = vector(1,nr);
	F2_12 = vector(1,nr);
	
	/* estimation of squared deviations */
	for(i=1;i<=nr;i++){
		F2_12[i] = est_f2(n, rs[i]);
		F_11[i] = sqrt(f2_11);
		F_22[i] = sqrt(f2_22);
//		fprintf(stderr,"%lf,%lf,%lf\n",F2_12[i],F_11[i],F_22[i]);
	}

	/* For measurement of processing time */
	end = clock();
	fprintf(stderr, "Finish time: %d\n", end);
	fprintf(stderr, "Processing time: %d[ms]\n", (end - start));

	/* output estimated results */
	for(i=1;i<=nr;i++){
		fprintf(stdout,"%ld,%lf,%lf,%lf,%lf,%lf,%lf\n",rs[i],F2_12[i],log((double)rs[i]/1.00)/log(10),log(fabs(F2_12[i]))/log(10)/2,F2_12[i]/(F_11[i]*F_22[i]),log(F_11[i])/log(10),log(F_22[i])/log(10));
//		fprintf(stdout,"%lf,%lf\n",log((double)rs[i])/log(10),log(F2[i])/log(10)/2);
	}

	/* Release allocated memory. */
	free_vector(data1,1,n);
	free_vector(y01,1,n);
	free_vector(data2,1,n);
	free_vector(y02,1,n);
	free_vector(F_11,1,nr);
	free_vector(F_22,1,nr);
	free_vector(F2_12,1,nr);

	exit(0);
}

double est_f2(long n, long scale){
	double f2,a0_1,a0_2;
	double y00_1,y0i_1,y10_1,y1i_1;
	double y00_2,y0i_2,y10_2,y1i_2;
	double temp1,temp2;
	long i,j; /* n: data length */
	long itemp;
	long k,ic;
	
	k = (scale-1)/2;
	ic = k+1;
	
	/* initial values */
	y10_1=0;
	y10_2=0;
	for(i=1;i<=scale;i++){
		y10_1 += y01[i];
		y10_2 += y02[i];
	}


	cmat((double)k);

	a0_1 = c1*y10_1;
	a0_2 = c1*y10_2;
	temp1 = (y01[ic] - a0_1);
	temp2 = (y02[ic] - a0_2);
	f2 = temp1*temp2;
	f2_11 = temp1*temp1;
	f2_22 = temp2*temp2;

	itemp = n-scale;
	
	for(i=1;i<=itemp;i++){
		if(i % i_refresh == 0){
			y10_1=0;
			y10_2=0;
			for(j=1;j<=scale;j++){
				y10_1 += y01[i+j];
				y10_2 += y02[i+j];
			}
		}else{
			y00_1 = y10_1;
			y10_1 = y00_1 + y01[i+scale]-y01[i];
			y00_2 = y10_2;
			y10_2 = y00_2 + y02[i+scale]-y02[i];
		}
		
		a0_1 = c1*y10_1;
		a0_2 = c1*y10_2;
		temp1 = (y01[i+ic] - a0_1);
		temp2 = (y02[i+ic] - a0_2);
		f2_11 += temp1*temp1;
		f2_22 += temp2*temp2;
		f2 += temp1*temp2;
	}

	f2_11 = f2_11/(double)(itemp+1);
	f2_22 = f2_22/(double)(itemp+1);

	return f2/(double)(itemp+1);
}


/* Read input data */
long read_data(char filename[], int skip, int selcol1, int selcol2){

    long maxdat;
    long n;
    long lag_max,j;
    long spc, c, i, frst;
    double *tmp1,*tmp2;
    double yave1,yave2;
    double ccf_max,ccf_pos,ccf_neg;
	char cdata1[64],cdata2[64];
	char buf[256];
	FILE *fp1;
		
    if((fp1 = fopen(filename,"r")) == NULL){
		fprintf(stderr,"File %s not found!\n",filename);
		return 0;
	}else{
		fprintf(stderr,"Input file: %s.\n",filename);
	}

	if(skip > 0){
	    fprintf(stderr,"***** Skip the first %d lines *****\n", skip);
		for(i=1;i<=skip;i++){
			if(fgets(buf,256,fp1)!=NULL){
			    fprintf(stderr,"%s",buf);
   		 	}
		}
	    fprintf(stderr,"***********************************\n", skip);
	}

	n = 0;
	spc = 0;
	frst = 0;
	strcpy_s(cdata1,64,"");
	strcpy_s(cdata2,64,"");
	
	maxdat=0;
	tmp1 = vector(1,maxdat);
	tmp2 = vector(1,maxdat);

	while((c = fgetc(fp1)) != EOF){
		if(frst == 0 && (c == 9 || c == 32)){
			;;;
		}else if(frst == 0 && c != 9 && c != 32 && c != 44){
			frst = 1;
			if(selcol1 == 1) strcat_s(cdata1,64,&c);
			if(selcol2 == 1) strcat_s(cdata2,64,&c);
		}else if(frst >= 1 && ((spc == 0 && (c == 9 || c == 32)) || c == 44)){
			spc = 1;
			frst++;
		}else if(frst >= 1 && spc == 1 && (c == 9 || c == 32)){
			;;;
		}else if(c == 10){
	        if (++n >= maxdat) {	    
				double *s1,*s2;
				maxdat += 10000;	/* allow the input buffer to grow (the
											   increment is arbitrary) */
				if ((s1 = realloc(tmp1, maxdat * sizeof(double))) == NULL) {
				fprintf(stderr,
					"%s: insufficient memory, truncating input at row %d\n",
					pname, n);
					break;
				}
				if ((s2 = realloc(tmp2, maxdat * sizeof(double))) == NULL) {
				fprintf(stderr,
					"%s: insufficient memory, truncating input at row %d\n",
					pname, n);
					break;
				}
				tmp1 = s1;
				tmp2 = s2;
			}
			
			
			if(isalpha(cdata1[0])){
				tmp1[n] = -1;
			}else{
				tmp1[n] = atof(cdata1);
			}
			if(isalpha(cdata2[0])){
				tmp2[n] = -1;
			}else{
				tmp2[n] = atof(cdata2);
			}
			strcpy_s(cdata1,64,"");
			strcpy_s(cdata2,64,"");
			frst = 0;
			spc = 0;
		}else{
			spc = 0;
			if(frst == selcol1) strcat_s(cdata1,64,&c);
			if(frst == selcol2) strcat_s(cdata2,64,&c);
		}
	}
	
	data1 = vector(1,n);
	data2 = vector(1,n);
	for(i=1;i<=n;i++){
		data1[i] = tmp1[i];
		data2[i] = tmp2[i];
	}
    
    free_vector(tmp1, 1,maxdat);
    free_vector(tmp2, 1,maxdat);
    fclose(fp1);
	if (n < 1){
		fprintf(stderr,"No data was read!\n");
	    return 0;
	}
	
	y01 = vector(1,n);
	y02 = vector(1,n);
	
	/* calulation of mean */
	yave1 = 0;
	yave2 = 0;
	for(i=1;i<=n;i++){
		yave1 += data1[i];
		yave2 += data2[i];
	}
	yave1 = yave1/(double)n;
	yave2 = yave2/(double)n;
	
	/* integration */

	fprintf(stderr,"Lag length: %ld\n",lag);
	n = n-fabs(lag);

	if(lag == 0){
		y01[1] = (data1[1]-yave1);
		y02[1] = (data2[1]-yave2);
		for(i=2;i<=n;i++){
			y01[i] = y01[i-1]+(data1[i]-yave1);
			y02[i] = y02[i-1]+(data2[i]-yave2);
		}
	}else if(lag > 0){
		y01[1] = (data1[1]-yave1);
		y02[1] = (data2[1+lag]-yave2);
		fprintf(stderr,"data1[i] x data2[i+%ld]\n",lag);
		for(i=2;i<=n;i++){
			y01[i] = y01[i-1]+(data1[i]-yave1);
			y02[i] = y02[i-1]+(data2[i+lag]-yave2);
		}
	}else{
		lag = -lag;
		y01[1] = (data1[1+lag]-yave1);
		y02[1] = (data2[1]-yave2);
		fprintf(stderr,"data1[i+%ld] x data2[i]\n",lag);
		for(i=2;i<=n;i++){
			y01[i] = y01[i-1]+(data1[i+lag]-yave1);
			y02[i] = y02[i-1]+(data2[i]-yave2);
		}
	}
	
	return n;
}

static char *help_strings[] = {
 "usage: %s [OPTIONS ...]\n",
 "where OPTIONS may include:",
 " -s SKIP          skipped lines (default: SKIP = 0)",
 " -c C1 C2         selected column (default: C1 = 1, C2 = 2)",
 " -L Lag           lag between two time series (default: Lag = 0)",
 " -l MINBOX        smallest box width (default: 5)",
 " -u MAXBOX        largest box width (default: (data length)/3)",
 " -h               print this usage summary ",
NULL
};

void help(void)
{
    int i;

    (void)fprintf(stderr, help_strings[0], pname);
    for (i = 1; help_strings[i] != NULL; i++)
	(void)fprintf(stderr, "%s\n", help_strings[i]);
}

/* This function is based on that of the same names in dfa.c. */
int rscale(long minbox, long maxbox, double boxratio)
{
    long ir, n;
    long rw;
	/*  */
    rslen = log10(maxbox /minbox) / log10(boxratio) + 1;
    rs = lvector(1, rslen);
    for (ir = 1, n = 2, rs[1] = minbox; n <= rslen && rs[n-1] < maxbox; ir++)
      if ((rw = minbox * pow(boxratio, ir) + 0.5) > rs[n-1])
            rs[n++] = (long)(rw/2)*2+1;
    if (rs[--n] > maxbox) --n;
    return (n);
}

/* calculation of coefficients */
void cmat(double k)
{
	c1 = 1/(2*k+1);
}

/* The functions below are based on those of the same names in Numerical
   Recipes (see above). */
void error(char error_text[])
{
    fprintf(stderr, "%s: %s\n", pname, error_text);
    exit(1);
}


double *vector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
    double *v = (double *)malloc((size_t)((nh-nl+2) * sizeof(double)));
    if (v == NULL) error("allocation failure in vector()");
    return (v-nl+1);
}

int *ivector(long nl, long nh)
/* allocate an int vector with subscript range v[nl..nh] */
{
    int *v = (int *)malloc((size_t)((nh-nl+2) * sizeof(int)));
    if (v == NULL) error("allocation failure in ivector()");
    return (v-nl+1);
}

long *lvector(long nl, long nh)
/* allocate a long int vector with subscript range v[nl..nh] */
{
    long *v = (long *)malloc((size_t)((nh-nl+2) * sizeof(long)));
    if (v == NULL) error("allocation failure in lvector()");
    return (v-nl+1);
}

void free_vector(double *v, long nl, long nh)
/* free a double vector allocated with vector() */
{
    free(v+nl-1);
}

void free_ivector(int *v, long nl, long nh)
/* free an int vector allocated with ivector() */
{
    free(v+nl-1);
}

void free_lvector(long *v, long nl, long nh)
/* free a long int vector allocated with lvector() */
{
    free(v+nl-1);
}

