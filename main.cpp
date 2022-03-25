#include <iostream>
#include <math.h>
#include <omp.h>

#define X_LEFT_BOUND 1
#define X_RIGHT_BOUND 2
#define Y_LOW_BOUND 2
#define Y_UP_BOUND 3
#define N_MAX 10000
double func(double x){
    return exp(cos(x)) / 10;
}

void solve_wave(int nx, int nt){
    int i, j;

    double x_zero = 0;
    double x_last = 1;
    double t_zero = 0;
    double t_last = 2;

    double dx = (x_last - x_zero) / (nx - 1);
    double dt = (t_last - t_zero) / (nt - 1);


    double grid[nx][nt];

    for(i = 0; i < nx; i++){
        for(j = 0; j < nt; j++){
            grid[i][j] = 0;
        }
    }

    for(j = 0; j < nt; j++){
        grid[0][j] = 0;
        grid[nx-1][j] = 0;
    }
    for(i = 0; i < nx; i++){
        grid[i][0] = 0.1 * sin(i * dx * M_PI);
        grid[i][1] = grid[i][0];
    }

    for(j = 1; j < nt-1; j++){
        for(i = 1; i < nx-1; i++){
            grid[i][j+1] = 2 * grid[i][j] - grid[i][j-1] + pow(dt/dx, 2) * func(i * dx)
                                                           * (grid[i+1][j] - 2 * grid[i][j] + grid[i-1][j]);
        }
    }

    FILE* fp;
    fp = std::fopen("gif_dataset.tmp", "w");

    for(j = 0; j < nt; j++){
        for(i = 0; i < nx; i++){
            fprintf(fp, "%f %f\n", i*dx, grid[i][j]);
        }
        if(j != nt-1) {
            fprintf(fp, "\n\n");
        }
    }
    std::fclose(fp);

    FILE* gnupipe = _popen("gnuplot -persistent", "w");
    fprintf(gnupipe, "cd \"C:\\\\Users\\\\Dinislam\\\\CLionProjects\\\\HPC_HW2\\\\cmake-build-debug\"\n");
    fprintf(gnupipe, "stats \"gif_dataset.tmp\" name \"A\"\n");
    fprintf(gnupipe, "set xrange[A_min_x:A_max_x]\n"
                     "set yrange[A_min_y-1:A_max_y+1]\n");

    fprintf(gnupipe, "set term gif animate delay 5 size 300, 300 crop\n  ");
    fprintf(gnupipe, "set output \"gif_HW2.gif\"\n");
    fprintf(gnupipe, "set title \"Animated\"\n");
    fprintf(gnupipe,"do for [i=0:int(A_blocks-1)] \\\n "
                    "{ plot \"gif_dataset.tmp\" \\\n"
                    "index i u 1:2 w lines title \"Animation\"}\n", nt, dx);

    fprintf(gnupipe, "exit");
    _pclose(gnupipe);
}

double solve_wave_omp(int nx, int nt, int n_threads){

    int i, j;

    double x_zero = 0;
    double x_last = 1;
    double t_zero = 0;
    double t_last = 2;

    double dx = (x_last - x_zero) / (nx - 1);
    double dt = (t_last - t_zero) / (nt - 1);

    double** grid = (double**)malloc(nx* sizeof(double*));
    for(int i = 0; i < nx; i++){
        grid[i] = (double*)malloc(nt* sizeof(double));
    }

    double start = omp_get_wtime();

    #pragma omp parallel shared(grid, nt) num_threads(n_threads)
    {
        #pragma omp for
        for (j = 0; j < nt; j++) {
            grid[0][j] = 0;
            grid[nx - 1][j] = 0;
        }
    }

    #pragma omp parallel shared(grid, nx) num_threads(n_threads)
    {
        #pragma omp for
        for (i = 0; i < nx; i++) {
            grid[i][0] = 0.1 * sin(i * dx * M_PI);
            grid[i][1] = grid[i][0];
        }
    }
    for(j = 1; j < nt-1; j++){
        #pragma omp parallel shared(grid, nx, dt, dx) num_threads(n_threads)
        {
            #pragma omp for
            for (i = 1; i < nx - 1; i++) {
                grid[i][j + 1] = 2 * grid[i][j] - grid[i][j - 1] + pow(dt / dx, 2) * func(i * dx)
                                                                   * (grid[i + 1][j] - 2 * grid[i][j] + grid[i - 1][j]);
            }
        }
    }
    for(int i = 0; i < nx; i++){
        free(grid[i]);
    }
    free(grid);
    return omp_get_wtime() - start;
}

void plot_performance_graph(int start_nx, int end_nx, int step_nx){
    /*
    FILE* fp;
    fp = std::fopen("task_1_dataset.tmp", "w");

    double time[4][(end_nx-start_nx)/step_nx];
    for(int n_th = 1; n_th <= 4; n_th++) {
        for (int i = 0; i <= (end_nx - start_nx) / step_nx; i++) {
            std::cout << i <<std::endl;
            time[n_th-1][i] = solve_wave_omp(start_nx + i * step_nx, 25000, pow(2, n_th));
            fprintf(fp, "%d %f\n", start_nx + i * step_nx, time[n_th-1][i]);
        }
        fprintf(fp, "\n\n");
    }
    */
    FILE* gnupipe = _popen("gnuplot -persistent", "w");
    fprintf(gnupipe, "cd \"C:\\\\Users\\\\Dinislam\\\\CLionProjects\\\\HPC_HW2\\\\cmake-build-debug\"\n");
    fprintf(gnupipe, "set xlabel \"Spacial steps\"\n"
                     "set ylabel \"Time\"\n"
                     "set title \"Multithread performance\"\n");
    fprintf(gnupipe, "plot \"task_1_dataset.tmp\" index 0 u 1:2 w lines title \"Threads = 1\", \\\n"
                     "\"task_1_dataset.tmp\" index 1 u 1:2 w lines title \"Threads = 2\", \\\n"
                     "\"task_1_dataset.tmp\" index 2 u 1:2 w lines title \"Threads = 4\", \\\n"
                     "\"task_1_dataset.tmp\" index 3 u 1:2 w lines title \"Threads =8 \"\n");

    fprintf(gnupipe, "exit");
    _pclose(gnupipe);

}

// Vector allocation
void AllocateVector(double** Vector, int size)
{
    (*Vector) = new double[size];
}
void AllocateVector(int** Vector, int size)
{
    (*Vector) = new int[size];
}
// Vector release
void FreeVector(double** Vector)
{
    delete[](*Vector);
}
void FreeVector(int** Vector)
{
    delete[](*Vector);
}

double GetWParam(double Step){
    return 1.3;
}

double f(double x, double y){
    return 4;
}

double FuncU(double x, double y){
    return x*x + y*y;
}

double mu1(double y){
    return 1 + y*y;
}

double mu2(double y){
    return 4 + y*y;
}
double mu3(double x){
    return x*x + 4;
}
double mu4(double x){
    return x*x + 9;
}
void CreateDUMatrix(int n, int m, double** Matrix, int** Index)
{
    double hsqr = (double)n * n / (X_RIGHT_BOUND-X_LEFT_BOUND) / (X_RIGHT_BOUND-X_LEFT_BOUND); // 1/h
    double ksqr = (double)m * m / (Y_UP_BOUND-Y_LOW_BOUND) / (Y_UP_BOUND-Y_LOW_BOUND); // 1/k
    double A = 2 * (hsqr + ksqr);
    int size = (n - 1) * (m - 1), bandWidth = 5;
    AllocateVector(Matrix, size * bandWidth);
    AllocateVector(Index, bandWidth);
    (*Index)[0] = -n + 1; (*Index)[1] = -1; (*Index)[2] = 0; (*Index)[3] = 1; (*Index)[4] = n - 1;
    for (int i = 0; i < size; i++)
    {
        if (i >= n - 1) (*Matrix)[i * bandWidth] = -ksqr;
        else (*Matrix)[i * bandWidth] = 0.0;
        if (i % (n - 1) != 0) (*Matrix)[i * bandWidth + 1] = -hsqr;
        else (*Matrix)[i * bandWidth + 1] = 0.0;
        (*Matrix)[i * bandWidth + 2] = A;
        if ((i + 1) % (n - 1) != 0) (*Matrix)[i * bandWidth + 3] = -hsqr;
        else (*Matrix)[i * bandWidth + 3] = 0.0;
        if (i < (n - 1) * (m - 2)) (*Matrix)[i * bandWidth + 4] = -ksqr;
        else (*Matrix)[i * bandWidth + 4] = 0.0;
    }
}


void CreateDUVector(int n, int m, double** Vector)
{
    double h = (X_RIGHT_BOUND-X_LEFT_BOUND) / (double)n;
    double k = (Y_UP_BOUND-Y_LOW_BOUND) / (double)m;
    double hsqr = (double)n * n / (X_RIGHT_BOUND-X_LEFT_BOUND) / (X_RIGHT_BOUND-X_LEFT_BOUND);
    double ksqr = (double)m * m / (Y_UP_BOUND-Y_LOW_BOUND) / (Y_UP_BOUND-Y_LOW_BOUND);
    AllocateVector(Vector, (n - 1) * (m - 1));
    for (int j = 0; j < m - 1; j++)
    {
        for (int i = 0; i < n - 1; i++)
            (*Vector)[j * (n - 1) + i] = f((double)(i + 1) * h, (double)(j + 1) * k);
        (*Vector)[j * (n - 1)] += hsqr * mu1(Y_LOW_BOUND + (double)(j + 1) * k);
        (*Vector)[j * (n - 1) + n - 2] += hsqr * mu2(Y_LOW_BOUND + (double)(j + 1) * k);
    }
    for (int i = 0; i < n - 1; i++)
    {
        (*Vector)[i] += ksqr * mu3(X_LEFT_BOUND + (double)(i + 1) * h);
        (*Vector)[(m - 2) * (n - 1) + i] += ksqr * mu4(X_LEFT_BOUND + (double)(i + 1) * h);
    }
}

void GetFirstApproximation(double* Result, int size)
{
    for (int i = 0; i < size; i++)
        Result[i] = 0.0;
}

double BandOverRelaxation(double* Matrix, double* Vector, double* Result, int* Index, int
size, int bandWidth,
                          double WParam, double Accuracy, int& StepCount)
{
    double CurrError;//achieved accuracy
    double sum, TempError;
    int ii, index = Index[bandWidth - 1], bandHalf = (bandWidth - 1) / 2;
    StepCount = 0;
    do
    {
        CurrError = -1.0;
        for (int i = index; i < size + index; i++)
        {
            ii = i - index;
            TempError = Result[i];
            sum = 0.0;
            for (int j = 0; j < bandWidth; j++)
                sum += Matrix[ii * bandWidth + j] * Result[i + Index[j]];
            Result[i] = (Vector[ii] - sum) * WParam / Matrix[ii * bandWidth + bandHalf] + Result[i];
            TempError = fabs(Result[i] - TempError);
            if (TempError > CurrError) CurrError = TempError;
        }
        StepCount++;
    } while ((CurrError > Accuracy) && (StepCount < N_MAX));
    return CurrError;
}


double SolvePoisson(int n, int m, double* Solution, double Accuracy, double& ORAccuracy,
                    int& StepCount)
{
    double* Matrix, * Vector, * Result;
    int* Index;
    int size = (n - 1) * (m - 1), ResSize = size + 2 * (n - 1), bandWidth = 5;
    double start, finish; double time;
    double WParam, step = (n / (X_RIGHT_BOUND-X_LEFT_BOUND) > m / (Y_UP_BOUND-Y_LOW_BOUND)) ?
                          (double)(X_RIGHT_BOUND-X_LEFT_BOUND) / n : (double)(Y_UP_BOUND-Y_LOW_BOUND) / m;
    CreateDUMatrix(n, m, &Matrix, &Index);
    CreateDUVector(n, m, &Vector);
    AllocateVector(&Result, ResSize);
    GetFirstApproximation(Result, ResSize);
    WParam = GetWParam(step);
    start = omp_get_wtime();
    ORAccuracy = BandOverRelaxation(Matrix, Vector, Result, Index, size, bandWidth,
                                    WParam, Accuracy, StepCount);
    finish = omp_get_wtime();
    time = (finish - start);
    memcpy(Solution, Result + n - 1, sizeof(double) * size);
    FreeVector(&Matrix);
    FreeVector(&Index);
    FreeVector(&Vector);
    FreeVector(&Result);
    return time;
}

double SolutionCheck(double* solution, int n, int m)
{
    double h = (X_RIGHT_BOUND-X_LEFT_BOUND) / (double)n, k = (Y_UP_BOUND-Y_LOW_BOUND) / (double)m;
    double err = 0, temp;
    for (int j = 0; j < m - 1; j++)
        for (int i = 0; i < n - 1; i++)
        {
            temp = fabs(solution[j * (n - 1) + i] - FuncU(X_LEFT_BOUND + (double)(i + 1) * h, Y_LOW_BOUND + (double)(j + 1) * k));
            if (temp > err)
                err = temp;
        }
    return err;
}


int main() {
    // Task 1 get a gif
    // solve_wave(100, 400);

    // Task 1 get chart
    // plot_performance_graph(2500, 10000, 500);

    // Task 2 build a table
    /*
    int dim[5] = {10, 50, 100, 500, 1000};
    double acc[3] = {0.001, 0.0001, 0.00001};
    int n, m;
    int StepCount;
    int size;
    double time;
    double Accuracy;
    double AcAccuracy;
    double Correctness;
    double* Solution;
    for(int i = 0; i < 5; i++) {
        n = m = dim[i];
        for(int j = 0; j < 3; j++) {
            Accuracy = acc[j];
            size = (n - 1) * (m - 1);
            AllocateVector(&Solution, size);
            time = SolvePoisson(n, m, Solution, Accuracy, AcAccuracy, StepCount);
            printf("N=M=%d\n", n);
            printf("Estimated Accuracy = %f\n", Accuracy);
            printf("UpperRelaxation:\ntime = %.15f\n", time);
            printf("Accuracy = %.15f, stepCount = %d\n", AcAccuracy, StepCount);
            Correctness = SolutionCheck(Solution, n, m);
            printf("Exact and Approximate solution comparison = %.15f\n", Correctness);
            FreeVector(&Solution);
        }
    }
     */

    // Task 2 plot a chart.
    /*
    int n=100;
    int m=100;
    int StepCount;
    int size;
    double time;
    double Accuracy = 0.0001;
    double AcAccuracy;
    double Correctness;
    double* Solution;
    size = (n - 1) * (m - 1);
    AllocateVector(&Solution, size);
    SolvePoisson(n, m, Solution, Accuracy, AcAccuracy, StepCount);

    double grid[n][m];
    FILE* fp2;
    fp2 = fopen("task_2_dataset.tmp", "w");

    for(int j = 0; j < m-1; j++){
        for(int i = 0; i < n-1; i++){
            fprintf(fp2, "%f %f %f\n", (double)X_LEFT_BOUND + i * (double)(X_RIGHT_BOUND-X_LEFT_BOUND)/n,
                    (double)Y_LOW_BOUND + j * (double)(Y_UP_BOUND-Y_LOW_BOUND)/m, Solution[i+j*(m-1)]);
        }
        fprintf(fp2, "\n");
    }
    FILE* gnupipe = _popen("gnuplot -persistent", "w");
    fprintf(gnupipe, "cd \"C:\\\\Users\\\\Dinislam\\\\CLionProjects\\\\HPC_HW2\\\\cmake-build-debug\"\n");
    fprintf(gnupipe, "set xlabel \"X\"\n"
                     "set ylabel \"Y\"\n"
                     "set title \"Heatmap\"\n");
    fprintf(gnupipe, "set size square\n");
    fprintf(gnupipe, "set pm3d map interpolate 0,0\n");
    fprintf(gnupipe, "splot \"task_2_dataset.tmp\" u 1:2:3 notitle\n");
    fprintf(gnupipe, "exit");
    _pclose(gnupipe);
     */
    return 0;
}
