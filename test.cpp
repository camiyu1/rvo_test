#include <glog/logging.h>
#include <iostream>
#include "tensor.hpp"

Tensor myTest() {
  Tensor result;
  LOG(INFO) << "In myTest result address : " << &result;
  return result;
}

Tensor wowTest(const Tensor& input) {
  LOG(INFO) << "enter wowTest()";
  Tensor result(input);
  LOG(INFO) << "return wowTest()";
  return result;
}


void test1() {
  std::vector<float> test{1};
/*  Tensor myTest; */
  Tensor another;
  Tensor abc(test, {1}, DataType::DT_FP32);
  another = abc;
  Tensor the_other = another;
  Tensor all_other = abc;
  LOG(INFO) << "Address of another   : " << std::hex << &another;
  LOG(INFO) << "Address of abc       : " << std::hex << &abc;
  LOG(INFO) << "Address of the_other : " << std::hex << &the_other;
  LOG(INFO) << "Address of all_other : " << std::hex << &all_other;
  Tensor rvo_test = myTest();
  LOG(INFO) << "Address of rvo_test  : " << std::hex << &rvo_test;
}

void test2() {
  std::vector<float> test{1};
  Tensor hello = Tensor(test, {1}, DataType::DT_FP32);
}

void test3() {
  std::vector<float> test{1};
  Tensor all_i_need_to_love(test, {1}, DataType::DT_FP32);
  all_i_need_to_love = wowTest(all_i_need_to_love);
}

int main(int argc, char** argv) {
  test3();
}