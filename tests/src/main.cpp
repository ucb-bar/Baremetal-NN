
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <rv.h>
#include <iostream>
// #include "nn.h"
// #include "unittest.h"

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

extern "C" void et_pal_init(void) {}

extern "C" void __dso_handle() {}

#define METHOD_ALLOCATOR_POOL_SIZE (70 * 1024 * 1024)
uint8_t* method_allocator_pool;

__ET_NORETURN void et_pal_abort(void) {
  __builtin_trap();
}


static uint8_t model_pte[] = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};


// DEFINE_string(
//     model_path,
//     "model.pte",
//     "Model serialized in flatbuffer format.");


using namespace torch::executor;
using torch::executor::util::BufferDataLoader;

int main() {
  runtime_init();

  method_allocator_pool = (uint8_t*)malloc(METHOD_ALLOCATOR_POOL_SIZE);


  auto loader = torch::executor::util::BufferDataLoader(model_pte, sizeof(model_pte));

  Result<torch::executor::Program> program =
      torch::executor::Program::load(&loader);

  std::cout << "hello world" << std::endl;

}