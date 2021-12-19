#include <limits.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define thread_num 8
#define ant_num 200
#define alpha  1
#define beta 3
#define evaporation_rate 0.3
#define total_run 100
#define total_pheromone_in_each_path 300000.0

typedef struct word //for read in file and tokenize it, then put into a node
{
    char *name;
    struct word *next;
}word;

typedef struct Queue // data structure
{
    word* front;
    word* end;
} Queue;

struct  // rank for mpi reduced
{
    int cost;
    int rank;
} loc_data, global_data;

int **dis; // the map that indicated that the distance between the node
int row, column;
int min_cost =INT_MAX; // global min cost for omp
int *min_path; // global min path
int id, comm_sz; // mpi id and mpi process number

int **alloc_memory(int ,int );
double **alloc_memory_f(int ,int );
double total_prob(double **, int *, int);
void ant_choose_path(double **, int **, int);
int read_file(char *);
void ant();
void enqueue(Queue *, char *);
word *dequeue(Queue *);

int main(int argc, char* argv[])
{
    int i;

    /* initial mpi information */
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* check the parameter */
    if(argc != 2)
    {
        printf("wrong total number of argv\n");
	return 0;
    }
    
    /* read in file */
    if(id == 0)
    {
        if(read_file(argv[1]))
            printf("Read file successfully!!\n");
	else
        {
            printf("Read file fails!!\n");
            return 0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    /* broadcast the map information to everyprocess */
    MPI_Bcast(&row, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&column, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* definitoin of global min path */
    min_path = malloc(sizeof(int) * column);

    /* definitoin of the map of the distance if process id is not 0 */
    if(id != 0)
    {
        dis = alloc_memory(row, column);
    }

    /* broadcast to every process the map information for the distance */
    for(i = 0; i < row; i++)
    {
        MPI_Bcast(dis[i], column, MPI_INT, 0, MPI_COMM_WORLD);
    }

    ant();
    
    MPI_Barrier(MPI_COMM_WORLD);

    /* compare the min cost with every process */
    loc_data.cost = min_cost;
    loc_data.rank = id;
    MPI_Allreduce(&loc_data, &global_data, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

    if (global_data.rank != 0) /* 0 already has the best tour */
    {
        if (id == 0)
        {
            MPI_Recv(min_path, column, MPI_INT, global_data.rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if  (id == global_data.rank)
        {
            MPI_Send(min_path, column, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    if(id == 0)
    {
        printf("min_cost = %d\n", global_data.cost);
        for( i = 0; i < column; i++)
        {
            printf("%3d", min_path[i]);
	}
        printf("\n");
    }

    free(dis[0]);
    free(dis);

    MPI_Finalize();

    return 0;
}

void ant()
{
#   pragma omp parallel num_threads(thread_num)
    {
        int i, t, k, length;
	double pheromone_in_each_path;
	int my_rank = omp_get_thread_num(); // thread id
        double **pheromone = alloc_memory_f(row, column);

	srand( id * comm_sz + my_rank * thread_num);

        // for initial pheromone matrix 
        for( i = 0 ; i < row ; i++)
        {
            for( t = 0 ; t < column ; t++)
            {
                pheromone[i][t] = 1.0;
            }
        }

        int **each_ant = alloc_memory(ant_num, column);

        // random choose each ant's start point
        for( k = 0 ; k < ant_num ; k++)
        {
            each_ant[k][0] = rand() % column;
            for( i = 1 ; i < column ; i++)
            {
                each_ant[k][i] = -1;
            }
        }

        for( t = 0 ; t < total_run ; t++)
        {

            /* choose the road for each ant in hole */
            for( k = 0 ; k < ant_num ; k++)
            {
                ant_choose_path(pheromone, each_ant, k);
            }

            /* decide the distance each ant goes, and check if the distance is min */
            for( k = 0 ; k < ant_num ; k++)
            {
                length = 0;
                for( i = 1 ; i < column ; i++)
                {
                    length += dis[each_ant[k][i-1]][each_ant[k][i]];
	        }
                length += dis[each_ant[k][column - 1]][each_ant[k][0]];
                if( length < min_cost) 
                {
#                   pragma omp critical
                    {
                        if( length < min_cost)
                        {
                            min_cost = length;
                            memcpy(min_path, each_ant[k], column * sizeof(int));
                        }
                    }
                }
            }
        
	    // updata pheromone matrix
            for( i = 0 ; i < row ; i++)
            {
                for( k = 0 ; k < column ; k++)
                {
                    pheromone[i][k] *= (1.0 - evaporation_rate); 
	        }
	    }
            pheromone_in_each_path = total_pheromone_in_each_path / (double) length;
            for( k = 0 ; k < ant_num ; k++)
            {
                for( i = 1 ; i < column ; i++)
                {
                    pheromone[each_ant[k][i-1]][each_ant[k][i]] += pheromone_in_each_path;
                    pheromone[each_ant[k][i]][each_ant[k][i-1]] += pheromone_in_each_path;
	        }
	    }
        }

        free(pheromone[0]);
        free(pheromone);
        free(each_ant[0]);
        free(each_ant);
    }
}

double total_prob(double **pheromone, int *step_check, int start)
{
    double prob = 0.0;
    int i = 0;
    for(; i < column ; i++ )
    {
        /* clculas the total sum of probility ant choose */
        if(step_check[i] == 0)
        {
            prob += (double)(pow(pheromone[start][i], alpha) * pow((double)1.0 / dis[start][i], beta));
        }
    }
    return prob;
}
void ant_choose_path(double **pheromone, int **path, int row_c)
{
    int *step_check = malloc(sizeof(int) * column); // the matrix for check if the ant has been gone yet
    int i, j, find;
    double prob;
    double total;
    double next;

    // init step_check
    memset( step_check, 0, sizeof( int ) * column );
    step_check[path[row_c][0]] = 1;

    for(i = 1 ; i < column ; i++)
    {
        prob = total_prob(pheromone, step_check, path[row_c][i-1] );
        next = (double) rand() /  (RAND_MAX + 1.0); // decide the porb num where to go
        j = -1;
	total = 0.0;
	find = 0;
	while((double)total / prob  < next && find == 0)
        {
            while((++j < column)&&step_check[j] == 1)
	    {
                 ;
	    }
            if(j == column)
            {
                while(step_check[--j] == 1) ;
                    find = 1;
            }
	    total +=(double) pow( pheromone[path[row_c][i-1]][j], alpha) * pow( (double)1.0 / dis[path[row_c][i-1]][j], beta);
        }
        path[row_c][i] = j;
        step_check[j] = 1;
    }
    free(step_check);
}

int read_file( char *filename)
{
    int i= 0, j = 0;
    char buffer[10000];
    char *token;
    word *temp;

    Queue *msg = malloc(sizeof(Queue));
    msg->front = msg->end = NULL;

    FILE *fp = fopen(filename, "r");
    if(!fp)
    {
        printf("Open file fails!!\n");
        return 0;
    }

    while(fgets(buffer, 9999, fp))
    {
        j = 0;
        token = strtok(buffer, " "); // get token
        while(token)
        {
            j++;
            enqueue(msg, token);
            token = strtok(NULL, " ");
        }
        i++;
    }
    i = j;
    row = i;
    column = j;

    dis = alloc_memory(row, column);

    // dequeue the node and put the token in the node to the map
    for(i = 0; i < row ; i++)
    {
        for(j = 0; j < column ; ++j)
        {
            temp = dequeue(msg);
            dis[i][j] = atoi(temp->name);
            free(temp);
	}
    }

    fclose(fp);
    free(msg);
    return 1;
}

void enqueue(Queue *q, char *token)
{
    word *temp = malloc(sizeof(word));
    temp->next = NULL;
    temp->name = malloc(strlen(token)+1);
    memcpy(temp->name, token, strlen(token));

    if(!q->front)
        q->front = q->end = temp;
    else
    {
        q->end->next = temp;
        q->end = temp;
    }
}
word *dequeue(Queue *q)
{
    word *temp;
    if(!q->front) temp = NULL;
    else if(q->front == q->end)
    {
        temp = q->front;
        q->front = q->end = NULL;
    }
    else
    {
        temp = q->front;
        q->front = q->front->next;
    }
    return temp;
}

double **alloc_memory_f(int row, int col)
{
    int i;
    double **temp = (double **)malloc( sizeof(double *)  * row );
    double *temp2 = (double *)malloc( sizeof(double) * row * col );
    memset( temp, 0, sizeof( double ) * row);
    memset( temp2, 0, sizeof( double ) * row * col );

    for( i = 0; i < row; i++)
    {
        temp[ i ] = &temp2[i*col];
    }
    return temp;
}

int **alloc_memory(int row,int col)
{
    int i;
    int **temp = (int **)malloc( sizeof(int *)  * row );
    int *temp2 = (int *)malloc( sizeof(int) * row * col );
    memset( temp, 0, sizeof( int ) * row);
    memset( temp2, 0, sizeof( int ) * row * col );

    for( i = 0; i < row; i++)
    {
        temp[ i ] = &temp2[i*col];
    }

    return temp;
}
