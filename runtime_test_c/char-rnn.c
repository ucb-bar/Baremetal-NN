#include "nn.h"
#include "model.h"


int input_size = 57;
int hidden_size = 8;
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

char *input_strs[] = {
  "sakura",
  "Schadenfreude",
  "Ni Hao",
};

int main() {
  printf("\n\n");

  Matrix output;
  NN_initMatrix(&output, 1, output_size);
  
  Matrix input;
  NN_initMatrix(&input, 1, input_size + hidden_size);
  
  Matrix hidden;
  NN_initMatrix(&hidden, 1, hidden_size);

  int index;

  for (int i=0; i<1; i+=1) {
    char *str = input_strs[i];

    memset(hidden.data, 0, hidden.rows * hidden.cols * sizeof(float));
    
    for (int j=1; j<strlen(str); j+=1) {
      encodeOneHot(&input, str[j]);
      NN_linear(&hidden, &i2h_weight_transposed, &i2h_bias, &input);

      forward(&output, &hidden, &input);
    }
    
    // printMatrix(&output);
    index = NN_argmax(&output);
    
    printf("\n> %s\n", str);
    printf("score: (");
    NN_printFloat(output.data[index], 2);
    printf("), predicted: (%d, %s)\n", index, categories[index]);
  }

  return 0;
}
