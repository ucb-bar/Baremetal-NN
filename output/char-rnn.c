#include <math.h>

#include "weights.h"
#include "network.h"

Matrix I2H_WEIGHT = {
  .rows = I2H_WEIGHT_T_ROWS,
  .cols = I2H_WEIGHT_T_COLS,
  .data = I2H_WEIGHT_DATA
};

Matrix I2H_BIAS = {
  .rows = I2H_BIAS_ROWS,
  .cols = I2H_BIAS_COLS,
  .data = I2H_BIAS_DATA
};

Matrix H2O_WEIGHT = {
  .rows = H2O_WEIGHT_T_ROWS,
  .cols = H2O_WEIGHT_T_COLS,
  .data = H2O_WEIGHT_DATA
};

Matrix H2O_BIAS = {
  .rows = H2O_BIAS_ROWS,
  .cols = H2O_BIAS_COLS,
  .data = H2O_BIAS_DATA
};


int input_size = 57;
int hidden_size = 32;
int output_size = 18;


const char *categories[] = {
  "Arabic", "Chinese", "Czech", "Dutch", "English", "French", "German", 
        "Greek", "Irish", "Italian", "Japanese", "Korean", "Polish", "Portuguese", 
        "Russian", "Scottish", "Spanish", "Vietnamese"
};


int len_mapping = 57;
char letter_mapping[] = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',' ','.',',',';','\''};




void encodeOneHot(Matrix *input, char c) {
  memset(input->data, 0, input->rows * input->cols * sizeof(float));
  for (int i=0; i<len_mapping; i+=1) {
    if (letter_mapping[i] == c) {
      input->data[i] = 1;
      return;
    }
  }
}





void forward(Matrix *output, char *input) {
  Matrix input_1;
  initMatrix(&input_1, 1, input_size);

  Matrix hidden;
  initMatrix(&hidden, 1, hidden_size);
  memset(hidden.data, 0, hidden_size * sizeof(float));


  Matrix layer_input;
  initMatrix(&layer_input, 1, input_size + hidden_size);
  Matrix layer_1_output;
  initMatrix(&layer_1_output, 1, hidden_size);  
  Matrix layer_2_output;
  initMatrix(&layer_2_output, 1, output_size);

  int i=0;
  while (input[i] != '\0') {
    encodeOneHot(&input_1, input[i]);
    
    concatenate(&layer_input, &input_1, &hidden);

    matmul(&layer_1_output, &layer_input, &I2H_WEIGHT);
    matadd(&layer_1_output, &layer_1_output, &I2H_BIAS);

    memcpy(hidden.data, layer_1_output.data, hidden_size * sizeof(float));

    matmul(&layer_2_output, &layer_1_output, &H2O_WEIGHT);
    matadd(&layer_2_output, &layer_2_output, &H2O_BIAS);

    logSoftmax(output, &layer_2_output);

    i += 1;
  }
}


char *inputs[] = {
  "sakura",
  "Vandroogenbroeck",
  "Xue Bu Dong Le!",
};

int main() {
  printf("\n\n");

  Matrix output;
  initMatrix(&output, 1, output_size);

  int index;

  for (int i=0; i<3; i+=1) {

    char *input = inputs[i];

    forward(&output, input);

    // printMatrix(&output);
    index = argmax(&output);
    
    printf("\n> %s\n", input);
    printf("score: (");
    printDouble(output.data[index], 2);
    printf("), predicted: (%d, %s)\n", index, categories[index]);
  }

  return 0;
}