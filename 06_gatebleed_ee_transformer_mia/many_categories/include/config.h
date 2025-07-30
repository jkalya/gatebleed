#ifndef CONFIG_H
#define CONFIG_H

//Global preprocessor directives
//#define TEST_UTILS
//#define TEST_ATTENTION
//#define PRINT_OUT_INIT_VECTORS
//#define PRINT_OUT_TEST_ATTENTION_FORWARD_OPERATION

//#define TEST_FEEDFORWARD
//#define DEBUG_PRINT_MAIN
//#define TEST_FEEDFORWARD_TRAIN
// Declare global variables (using extern)
extern float GLOBAL_LEAKY_SLOPE;
extern float GLOBAL_learning_rate;
extern float GLOBAL_momentum ;
extern float GLOBAL_ATTENTION_learning_rate;
extern float GLOBAL_ATTENTION_momentum ;
extern const float GLOBAL_CONST_learning_rate;
extern const float GLOBAL_CONST_momentum ;
#endif // CONFIG_H
