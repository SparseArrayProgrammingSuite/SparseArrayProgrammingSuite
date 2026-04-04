#include <ctf.hpp>

using namespace CTF;

int ccsdt_map_test(int   n,
                   World &dw){

  int shapeNS6[] = {NS,NS,NS,NS,NS,NS};
  int nnnnnn[] = {n,n,n,n,n,n};
  int shapeNS4[] = {NS,NS,NS,NS};
  int nnnn[] = {n,n,n,n};

  Tensor<> W(4, nnnn, shapeNS4, dw, "W", 1);
  Tensor<> T(4, nnnn, shapeNS4, dw, "T", 1);
  Tensor<> Z(6, nnnnnn, shapeNS6, dw, "Z", 1);

  Z["hijmno"] += W["hijk"]*T["kmno"];

  return 1;
}

char* getCmdOption(char ** begin, char ** end, const std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) return *itr;
  return 0;
}

int main(int argc, char ** argv){
  int rank, np, n;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 4;
  } else n = 4;

  {
    World dw(argc, argv);
    int pass = ccsdt_map_test(n, dw);
    assert(pass);
    if (rank == 0) printf("ccsdt_map_test passed\n");
  }

  MPI_Finalize();
  return 0;
}
