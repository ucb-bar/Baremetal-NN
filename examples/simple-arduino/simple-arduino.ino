#include "nn.h"
#include "model.h"


Model model;

void setup() {
  Serial.begin(115200);

  model_init(&model);
}

void loop() {
  uint32_t start_time = micros();
  model_forward(&model);
  uint32_t end_time = micros();

  Serial.print(end_time - start_time);
  Serial.println("us");

  delay(1000);
}
