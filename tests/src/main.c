
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <rv.h>

#include "nn.h"

#ifdef RVV
  #include "riscv_vector.h"
#endif

static void enable_vector_operations() {
  #ifdef RVV
    unsigned long mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= 0x00000600 | 0x00006000 | 0x00018000;
    asm volatile("csrw mstatus, %0"::"r"(mstatus));
  #endif
}

uint8_t float_eq(float golden, float actual, float relErr) {
  return (fabs(actual - golden) < relErr) || (fabs((actual - golden) / actual) < relErr);
}

uint8_t compare(Tensor *golden, Tensor *actual) {
  for (size_t i = 0; i < golden->size; i += 1) {
    if (!float_eq(((float *)golden->data)[i], ((float *)actual->data)[i], 1e-6)) {
      return 0;
    }
  }
  return 1;
}

int main() {
  enable_vector_operations();

  size_t cycles;

  
  {
    printf("add:                    ");

    Tensor *a = NN_tensor(2, (size_t[]){ 3, 3 }, DTYPE_F32, (float[]){ 0.4962566, 0.7682218, 0.08847743, 0.13203049, 0.30742282, 0.6340787, 0.4900934, 0.89644474, 0.45562798 });

    Tensor *b = NN_tensor(2, (size_t[]){ 3, 3 }, DTYPE_F32, (float[]){ 0.6323063, 0.34889346, 0.4017173, 0.022325754, 0.16885895, 0.29388845, 0.5185218, 0.6976676, 0.8000114 });

    
    Tensor *golden = NN_tensor(2, (size_t[]){ 3, 3 }, DTYPE_F32, (float[]){ 1.1285629, 1.1171153, 0.49019474, 0.15435624, 0.47628176, 0.92796713, 1.0086153, 1.5941124, 1.2556393 });
    Tensor *actual = NN_zeros(2, (size_t[]){ 3, 3 }, DTYPE_F32);

    cycles = READ_CSR("mcycle");
    NN_add(actual, a, b);
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s  (%lu cycles)\n", compare(golden, actual) ? "PASS" : "FAIL", cycles);


    NN_deleteTensor(a);
    NN_deleteTensor(b);
    NN_deleteTensor(golden);
    NN_freeTensorData(actual);
    NN_deleteTensor(actual);
  }

  {
    printf("add1_F32:               ");

    Tensor *a = NN_tensor(2, (size_t[]){ 3, 3 }, DTYPE_F32, (float[]){ 0.16102946, 0.28226858, 0.68160856, 0.915194, 0.3970999, 0.8741559, 0.41940832, 0.55290705, 0.9527381 });

    float v = 0.8444218515250481;
    
    Tensor *golden = NN_tensor(2, (size_t[]){ 3, 3 }, DTYPE_F32, (float[]){ 1.0054513, 1.1266904, 1.5260304, 1.7596159, 1.2415218, 1.7185777, 1.2638302, 1.3973289, 1.7971599 });
    Tensor *actual = NN_zeros(2, (size_t[]){ 3, 3 }, DTYPE_F32);

    cycles = READ_CSR("mcycle");
    NN_add1_F32(actual, a, v);
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s  (%lu cycles)\n", compare(golden, actual) ? "PASS" : "FAIL", cycles);


    NN_deleteTensor(a);
    NN_deleteTensor(golden);
    NN_freeTensorData(actual);
    NN_deleteTensor(actual);
  }

  {
    printf("sub:                    ");

    Tensor *a = NN_tensor(2, (size_t[]){ 3, 3 }, DTYPE_F32, (float[]){ 0.03616482, 0.18523103, 0.37341738, 0.30510002, 0.9320004, 0.17591017, 0.26983356, 0.15067977, 0.031719506 });

    Tensor *b = NN_tensor(2, (size_t[]){ 3, 3 }, DTYPE_F32, (float[]){ 0.20812976, 0.929799, 0.7231092, 0.7423363, 0.5262958, 0.24365824, 0.58459234, 0.03315264, 0.13871688 });

    
    Tensor *golden = NN_tensor(2, (size_t[]){ 3, 3 }, DTYPE_F32, (float[]){ -0.17196494, -0.744568, -0.3496918, -0.43723625, 0.40570462, -0.06774807, -0.31475878, 0.11752713, -0.10699737 });
    Tensor *actual = NN_zeros(2, (size_t[]){ 3, 3 }, DTYPE_F32);

    cycles = READ_CSR("mcycle");
    NN_sub(actual, a, b);
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s  (%lu cycles)\n", compare(golden, actual) ? "PASS" : "FAIL", cycles);


    NN_deleteTensor(a);
    NN_deleteTensor(b);
    NN_deleteTensor(golden);
    NN_freeTensorData(actual);
    NN_deleteTensor(actual);
  }

  {
    printf("addInplace:             ");

    Tensor *b = NN_tensor(2, (size_t[]){ 3, 3 }, DTYPE_F32, (float[]){ 0.242235, 0.81546897, 0.7931606, 0.27825248, 0.4819588, 0.81978035, 0.99706656, 0.6984411, 0.5675464 });

    
    Tensor *golden = NN_tensor(2, (size_t[]){ 3, 3 }, DTYPE_F32, (float[]){ 0.242235, 0.81546897, 0.7931606, 0.27825248, 0.4819588, 0.81978035, 0.99706656, 0.6984411, 0.5675464 });
    Tensor *actual = NN_zeros(2, (size_t[]){ 3, 3 }, DTYPE_F32);

    cycles = READ_CSR("mcycle");
    NN_addInplace(actual, b);
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s  (%lu cycles)\n", compare(golden, actual) ? "PASS" : "FAIL", cycles);


    NN_deleteTensor(b);
    NN_deleteTensor(golden);
    NN_freeTensorData(actual);
    NN_deleteTensor(actual);
  }

  {
    printf("fill_F32:               ");

    float v = 0.7579544029403025;
    
    Tensor *golden = NN_tensor(2, (size_t[]){ 3, 3 }, DTYPE_F32, (float[]){ 0.7579544, 0.7579544, 0.7579544, 0.7579544, 0.7579544, 0.7579544, 0.7579544, 0.7579544, 0.7579544 });
    Tensor *actual = NN_zeros(2, (size_t[]){ 3, 3 }, DTYPE_F32);

    cycles = READ_CSR("mcycle");
    NN_fill_F32(actual, v);
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s  (%lu cycles)\n", compare(golden, actual) ? "PASS" : "FAIL", cycles);


    NN_deleteTensor(golden);
    NN_freeTensorData(actual);
    NN_deleteTensor(actual);
  }

  {
    printf("matmulT:                ");

    Tensor *a = NN_tensor(2, (size_t[]){ 6, 3 }, DTYPE_F32, (float[]){ 0.83524317, 0.20559883, 0.593172, 0.112347245, 0.15345693, 0.24170822, 0.7262365, 0.7010802, 0.20382375, 0.65105355, 0.774486, 0.43689132, 0.5190908, 0.61585236, 0.8101883, 0.98009706, 0.11468822, 0.31676513 });

    Tensor *b = NN_tensor(2, (size_t[]){ 5, 3 }, DTYPE_F32, (float[]){ 0.69650495, 0.9142747, 0.93510365, 0.9411784, 0.5995073, 0.06520867, 0.54599625, 0.18719733, 0.034022927, 0.94424623, 0.8801799, 0.0012360215, 0.593586, 0.41577, 0.41771942 });

    
    Tensor *golden = NN_tensor(2, (size_t[]){ 6, 5 }, DTYPE_F32, (float[]){ 1.3244021, 0.94805074, 0.51470864, 0.9703723, 0.82904994, 0.44457445, 0.21349882, 0.098291524, 0.2414519, 0.23145676, 1.3374035, 1.1171119, 0.5346975, 1.3030747, 0.8077131, 1.5700936, 1.1055566, 0.5153189, 1.2969818, 0.8909623, 1.6822176, 0.91059625, 0.42627248, 1.0332117, 0.90260935, 1.0837072, 1.0118583, 0.56737596, 1.0267907, 0.7617748 });
    Tensor *actual = NN_zeros(2, (size_t[]){ 6, 5 }, DTYPE_F32);

    cycles = READ_CSR("mcycle");
    NN_matmulT(actual, a, b);
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s  (%lu cycles)\n", compare(golden, actual) ? "PASS" : "FAIL", cycles);


    NN_deleteTensor(a);
    NN_deleteTensor(b);
    NN_deleteTensor(golden);
    NN_freeTensorData(actual);
    NN_deleteTensor(actual);
  }

  {
    printf("matmul:                 ");

    Tensor *a = NN_tensor(2, (size_t[]){ 6, 3 }, DTYPE_F32, (float[]){ 0.27112156, 0.6922781, 0.20384824, 0.68329567, 0.75285405, 0.8579358, 0.6869556, 0.005132377, 0.17565155, 0.7496575, 0.6046507, 0.10995799, 0.21209025, 0.97037464, 0.83690894, 0.28198743, 0.3741576, 0.023700953 });

    Tensor *b = NN_tensor(2, (size_t[]){ 3, 5 }, DTYPE_F32, (float[]){ 0.49101293, 0.123470545, 0.11432165, 0.4724502, 0.5750725, 0.29523486, 0.7966888, 0.19573045, 0.95368505, 0.84264994, 0.07835853, 0.37555784, 0.5225613, 0.57295054, 0.61858714 });

    
    Tensor *golden = NN_tensor(2, (size_t[]){ 6, 5 }, DTYPE_F32, (float[]){ 0.35348204, 0.6615625, 0.27301818, 0.90510166, 0.86536056, 0.6250023, 1.0063617, 0.673796, 1.5323635, 1.5580451, 0.3525831, 0.15487501, 0.17132716, 0.4300866, 0.5080299, 0.5552216, 0.61557466, 0.26151043, 0.99382263, 1.0086348, 0.45620644, 1.1135812, 0.6515146, 1.5051413, 1.4573545, 0.250781, 0.3418054, 0.11785651, 0.503633, 0.49210823 });
    Tensor *actual = NN_zeros(2, (size_t[]){ 6, 5 }, DTYPE_F32);

    cycles = READ_CSR("mcycle");
    NN_matmul(actual, a, b);
    cycles = READ_CSR("mcycle") - cycles;
    printf("%s  (%lu cycles)\n", compare(golden, actual) ? "PASS" : "FAIL", cycles);


    NN_deleteTensor(a);
    NN_deleteTensor(b);
    NN_deleteTensor(golden);
    NN_freeTensorData(actual);
    NN_deleteTensor(actual);
  }

}