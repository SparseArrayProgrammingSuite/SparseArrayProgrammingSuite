#include <ctf.hpp>

using namespace CTF;

int ccsdt_t3_to_t2(int     n,
                   int     m,
                   World & dw){

  int shapeAS4[] = {AS,NS,AS,NS};
  int shapeAS6[] = {AS,AS,NS,AS,AS,NS};
  int shapeTS6[] = {AS,NS,NS,NS,NS,NS};
  int shapeTS4[] = {AS,NS,NS,NS};
  int shapeNS4[] = {NS,NS,NS,NS};
  int nnnm[] = {n,n,n,m};
  int mmnn[] = {m,m,n,n};
  int mmmnnn[] = {m,m,m,n,n,n};

  Tensor<> AS_A(4, nnnm, shapeTS4, dw, "AS_A", 1);
  Tensor<> AS_B(6, mmmnnn, shapeAS6, dw, "AS_B", 1);
  Tensor<> AS_C(4, mmnn, shapeAS4, dw, "AS_C", 1);
  Tensor<> NS_A(4, nnnm, shapeNS4, dw, "NS_A", 1);
  Tensor<> NS_B(6, mmmnnn, shapeTS6, dw, "NS_B", 1);
  Tensor<> NS_C(4, mmnn, shapeTS4, dw);

  srand48(2013);
  AS_A.fill_random(0.,1.);
  AS_B.fill_random(0.,1.);
  AS_C.fill_random(0.,1.);

  NS_A["abij"] = AS_A["abij"];
  NS_B["abcijk"] = AS_B["abcijk"];
  NS_C["abij"] += AS_C["abij"];

  AS_C["abij"] += 0.5*AS_A["mnje"]*AS_B["abeimn"];

  NS_C["abij"] += 0.5*NS_A["mnje"]*NS_B["abeimn"];
  NS_C["abji"] -= 0.5*NS_A["mnje"]*NS_B["abeimn"];

  int pass = 1;

  double nrm_AS = sqrt((double)(AS_C["ijkl"]*AS_C["ijkl"]));
  double nrm_NS = sqrt((double)(NS_C["ijkl"]*NS_C["ijkl"]));
  double cnrm_AS = AS_C.norm2();
  double cnrm_NS = NS_C.norm2();
  if (fabs(nrm_AS-cnrm_AS) >= 1.E-6) { printf("ERROR: AS norm not working!\n"); pass = 0; }
  if (fabs(nrm_NS-cnrm_NS) >= 1.E-6) { printf("ERROR: NS norm not working!\n"); pass = 0; }

  NS_C["abij"] -= AS_C["abij"];
  double nrm = NS_C.norm2();
  if (fabs(nrm) > 1.E-6) pass = 0;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0){
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass)
      printf("{ AS_C[\"abij\"] += 0.5*AS_A[\"mnje\"]*AS_B[\"abeimn\"] } passed\n");
    else
      printf("{ AS_C[\"abij\"] += 0.5*AS_A[\"mnje\"]*AS_B[\"abeimn\"] } failed\n");
  } else
    MPI_Reduce(&pass, MPI_IN_PLACE, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

  return pass;
}

char* getCmdOption(char ** begin, char ** end, const std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) return *itr;
  return 0;
}

int main(int argc, char ** argv){
  int rank, np, n, m;
  int in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 6;
  } else n = 6;
  if (getCmdOption(input_str, input_str+in_num, "-m")){
    m = atoi(getCmdOption(input_str, input_str+in_num, "-m"));
    if (m < 0) m = 7;
  } else m = 7;

  {
    World dw(argc, argv);
    int pass = ccsdt_t3_to_t2(n, m, dw);
    assert(pass);
  }

  MPI_Finalize();
  return 0;
}
