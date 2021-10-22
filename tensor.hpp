/**
 * @file   tensor.hpp
 * @author Yong Hak Lee (camiyu1@gmail.com)
 * @brief  Header only N-dimensional Tensor Class
 * @date   Jun. 2021
 */

#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <type_traits>
#include <typeinfo>
#include <vector>

enum class DataType {
  DT_SINT4,
  DT_SINT8,
  DT_SINT16,
  DT_SMIX48,
  DT_UINT4,
  DT_UINT8,
  DT_UINT16,
  DT_UMIX48,
  DT_FP8,
  DT_FP16,
  DT_FP32,
  DT_END
};

typedef std::vector<int> shape_t;

/**
 * @brief N-dimensional Tensor Class
 */

class Tensor {
 public:
  enum class Format { NWHC, NHWC, NCHW, END };

  /**
   * @brief Destroy the Tensor object
   */
  virtual ~Tensor() {
    // Free pointer
    if (this->data_ != nullptr) {
      std::free(this->data_);
      this->data_ = nullptr;
    }
    LOG(INFO) << "Default destructor called";
  }

#if 0
  Tensor(Tensor const& other)
    : format_(other.format_), shape_(other.shape_), data_(other.data_),
    datatype_(other.datatype_) {
    LOG(INFO) << "Copy constructor called";
  }
#endif

  /**
   * @brief Construct a new Tensor object
   *
   * @param shape : Tensor shape
   * @param datatype : DT_SINT8, DT_UINT8, DT_FP32, ...
   */
  Tensor(const shape_t& shape, DataType datatype = DataType::DT_FP32,
         Tensor::Format format = Tensor::Format::NWHC)
      : format_(format), shape_(shape), datatype_(datatype), data_(nullptr) {
    setNWHC(format_);
    size_t size = GetTensorSize();
    elembytes_ = getElemBytes(datatype_);
    data_ = static_cast<u_char*>(std::calloc(size, elembytes_));
  }

  Tensor()
      : format_(Tensor::Format::NWHC),
        shape_(0),
        datatype_(DataType::DT_END),
        data_(nullptr) {
  LOG(INFO) << "Basic instructor called";
}

  // TODO(camiyu1): use factory pattern, constructor cannot get template
  template <typename T>
  Tensor(const std::vector<T>& ref_vector, const shape_t& shape,
         DataType datatype = DataType::DT_FP32,
         Tensor::Format format = Tensor::Format::NWHC)
      : format_(format), shape_(shape), datatype_(datatype) {
    setNWHC(format_);
    LOG(INFO) << "Ctor using ref_vector called";
    int size = GetTensorSize();
    elembytes_ = getElemBytes(datatype_);
    data_ = static_cast<u_char*>(std::calloc(size, elembytes_));
    SetData(ref_vector);
  }

  // when pt_size < tensor_size, zero padded
  // when pt_size >= tensor_size, only valid data used
  Tensor(const u_char* pt_src, const size_t pt_size, const shape_t& shape,
         DataType datatype = DataType::DT_FP32,
         Tensor::Format format = Tensor::Format::NWHC)
      : format_(format), shape_(shape), datatype_(datatype) {
    setNWHC(format_);
    int size = GetTensorSize();
    elembytes_ = getElemBytes(datatype_);
    data_ = static_cast<u_char*>(std::calloc(size, elembytes_));
    size_t tensor_size = size * elembytes_;
    size_t valid_size = std::min(tensor_size, pt_size);
    std::memcpy(data_, pt_src, valid_size);
    size_t remained_size = std::max(tensor_size - valid_size, 0LU);
    std::memset(data_ + valid_size, 0, remained_size);
  }

  Tensor(const Tensor& other)
      : format_(other.format_),
        shape_(other.shape_),
        datatype_(other.datatype_),
        elembytes_(other.elembytes_) {
    setNWHC(format_);
    int size = GetTensorSize();
    data_ = static_cast<u_char*>(std::calloc(size, elembytes_));
    std::memcpy(data_, other.data_, size * elembytes_);
    LOG(INFO) << "Copy constructor called";
  }

  template <typename T>
  Tensor& SetData(const std::vector<T>& ref_vector) {
    size_t size = std::min(ref_vector.size(), GetTensorSize());
    // Initialize the data with ref_vector
    for (size_t i = 0; i < size; ++i) {
      this->at<T>(i) = ref_vector[i];
    }
    // Initialize the remained data to 0
    for (size_t i = size; i < GetTensorSize(); ++i) {
      this->at<T>(i) = 0;
    }
    return *this;
  }

  template <typename T>
  T& at(int idx) const {
    return *(reinterpret_cast<T*>(data_ + idx * elembytes_));
  }

  template <typename T>
  std::vector<T> ToVector() const {
    int size = GetTensorSize();
    std::vector<T> output(size);

    for (int i = 0; i < size; ++i) {
      output.at(i) = this->at<T>(i);
    }
    return output;
  }

  template <typename T>
  T* data() {
    return (reinterpret_cast<T*>(data_));
  }

  template <typename T>
  T* data() const {
    return (reinterpret_cast<T*>(data_));
  }

  Tensor::Format GetFormat() const { return format_; }

  size_t GetTensorSize() const {
    int size = 1;
    for (auto shape_elem : shape_) {
      size *= shape_elem;
    }
    return size;
  }

  size_t GetTensorByteSize() const {
    return GetTensorSize() * getElemBytes(datatype_);
  }

  const shape_t& GetShape() const { return shape_; }

  int GetShape(int idx) const { return shape_[idx]; }

  DataType GetDataType() const { return datatype_; }

  std::vector<Tensor> SplitBatch(int newN) const {
    if (newN == 0) {
      LOG(ERROR) << "Cannot split tensor into zero batch";
      return std::vector<Tensor>();
    }
    int split_cnt = (GetBatch() + newN - 1) / newN;
    std::vector<Tensor> tensors;
    for (int bidx = 0; bidx < split_cnt; ++bidx) {
      size_t astride = GetWidth() * GetHeight() * GetChannels() * elembytes_;
      size_t gstride = astride * newN;
      size_t remain_batch =
          std::min(newN, std::max(0, GetBatch() - bidx * newN));
      unsigned begin = gstride * bidx;
      uint8_t* ptr = reinterpret_cast<uint8_t*>(data_) + begin;
      size_t size = astride * remain_batch;
      shape_t new_shape = shape_;
      new_shape[idx_n_] = newN;
      tensors.emplace_back(ptr, size, new_shape, datatype_, format_);
    }
    return tensors;
  }

  static Tensor MergeBatch(const std::vector<Tensor>& tensors,
                           std::optional<unsigned> valid_batch = std::nullopt) {
    // ===begin of condition check===
    // tensor should have at least one tensor
    if (tensors.size() == 0) {
      LOG(ERROR) << "the number of input tensors MUST be larger than 1";
      return Tensor();
    }
    if (valid_batch && *valid_batch == 0) {
      LOG(ERROR) << "the number of valid_batch MUST be larger than 0";
      return Tensor();
    }
    // tensor should have all same astride and format
    const Tensor& ltensor = tensors.front();
    const size_t lastride = ltensor.GetTensorSize() / ltensor.GetBatch();
    const auto lformat = ltensor.GetFormat();
    const auto ldt = ltensor.GetDataType();
    for (const Tensor& tensor : tensors) {
      size_t astride = tensor.GetTensorSize() / tensor.GetBatch();
      auto format = tensor.GetFormat();
      auto dt = tensor.GetDataType();
      if (astride != lastride || format != lformat || dt != ldt) {
        LOG(ERROR) << "Cannot merge inconsistent tensors";
        return Tensor();
      }
    }
    // ===end of condition check===
    // 1. get final shape
    size_t total_batch = 0;
    for (const Tensor& tensor : tensors) {
      total_batch += tensor.GetBatch();
    }
    auto final_shape = ltensor.GetShape();
    final_shape[0] =
        (valid_batch == std::nullopt)
            ? total_batch
            : std::min(static_cast<size_t>(*valid_batch), total_batch);
    size_t vbatch = final_shape[0];
    // 1-2. prepare final_tensor
    Tensor final_tensor(final_shape, ltensor.GetDataType(),
                        ltensor.GetFormat());
    uint8_t* ptr = final_tensor.data<uint8_t>();
    // 2. concat all batches
    for (const Tensor& tensor : tensors) {
      size_t vpartial_batch =
          std::min(static_cast<size_t>(tensor.GetBatch()), vbatch);
      vbatch -= vpartial_batch;

      /* byte size calculation */
      size_t bsize_per_batch = tensor.GetTensorByteSize() / tensor.GetBatch();
      size_t bsize_valid = bsize_per_batch * vpartial_batch;
      std::copy(tensor.data<uint8_t>(), tensor.data<uint8_t>() + bsize_valid,
                ptr);
      ptr += bsize_valid;
    }
    return final_tensor;
  }

  // Operator overloading
  bool operator==(const Tensor& other) const {
    if ((this->shape_ != other.shape_) ||
        (this->datatype_ != other.datatype_) ||
        (this->elembytes_ != other.elembytes_) ||
        (this->format_ != other.format_)) {
      return false;
    }

    return (std::memcmp(this->data_, other.data_,
                        other.GetTensorSize() * other.elembytes_) == 0);
  }

  Tensor& operator=(const Tensor& other) {
    LOG(INFO) << "operator=(const Tensor& other) called";
    if (this == &other) {
      LOG(INFO) << "- same";
      return *this;
    }
    this->shape_ = other.shape_;
    this->format_ = other.format_;

    int size = other.GetTensorSize();

    this->elembytes_ = other.elembytes_;
    this->datatype_ = other.datatype_;

    if (this->data_ != nullptr) std::free(this->data_);
    this->data_ = static_cast<u_char*>(std::calloc(size, elembytes_));
    std::copy(other.data_, other.data_ + size * other.elembytes_, this->data_);
    return *this;
  }

  // NWHC specific function
  int N() const { return GetShape(idx_n_); }
  int W() const { return GetShape(idx_w_); }
  int H() const { return GetShape(idx_h_); }
  int C() const { return GetShape(idx_c_); }

  // The following will be deprecated in the future version
  int GetBatch() const { return N(); }
  int GetWidth() const { return W(); }
  int GetHeight() const { return H(); }
  int GetChannels() const { return C(); }

 private:
  Tensor::Format format_ = Tensor::Format::NWHC;
  int idx_n_ = 0, idx_w_ = 1, idx_h_ = 2, idx_c_ = 3;

  Tensor& createTensor(const shape_t& shape, DataType datatype) {
    shape_ = shape;
    datatype_ = datatype;
    elembytes_ = getElemBytes(datatype_);
    int size = GetTensorSize();
    this->data_ = static_cast<u_char*>(std::calloc(size, elembytes_));
    return *this;
  }

  void setNWHC(Tensor::Format format) {
    switch (format) {  // Shape IDX now as NWHC fixed!!
      case Tensor::Format::NWHC:
        idx_n_ = 0;
        idx_w_ = 1;
        idx_h_ = 2;
        idx_c_ = 3;
        break;
      case Tensor::Format::NHWC:
        idx_n_ = 0;
        idx_w_ = 1;
        idx_h_ = 2;
        idx_c_ = 3;
        break;
      case Tensor::Format::NCHW:
        idx_n_ = 0;
        idx_w_ = 1;
        idx_h_ = 2;
        idx_c_ = 3;
        break;
      default:
        LOG(ERROR) << "Tensor format NYI";
    }
  }

  int getElemBytes(DataType datatype) const {
    int elembytes;
    switch (datatype) {
      case DataType::DT_FP8:
      case DataType::DT_SINT8:
      case DataType::DT_UINT8:
        elembytes = 1;
        break;
      case DataType::DT_SINT16:
      case DataType::DT_UINT16:
      case DataType::DT_FP16:
        elembytes = 2;
        break;
      case DataType::DT_FP32:
        elembytes = 4;
        break;
      default:
        elembytes = 1;
    }
    return elembytes;
  }

 protected:
  shape_t shape_;
  DataType datatype_;
  int elembytes_;
  u_char* data_;
};

/**
 * @brief Template Trait class for data types
 *
 * @tparam T Data Types of Cpp (float, uint8_t, int8_t, ...)
 */
template <typename T>
class Type {
 public:
};

template <>
class Type<float> {
 public:
  static const DataType datatype = DataType::DT_FP32;
};

template <>
class Type<int8_t> {
 public:
  static const DataType datatype = DataType::DT_SINT8;
};

template <>
class Type<uint8_t> {
 public:
  static const DataType datatype = DataType::DT_UINT8;
};

template <>
class Type<int16_t> {
 public:
  static const DataType datatype = DataType::DT_SINT16;
};

template <>
class Type<uint16_t> {
 public:
  static const DataType datatype = DataType::DT_UINT16;
};
template <>
class Type<uint64_t> {
 public:
  static const DataType datatype = DataType::DT_END;
};
//------------------------------------------------------------------------------

#if 0
/**
 * @brief TensorNWHC is a specific 4D float type tensor class for users
 */
class TensorNWHC : public Tensor {
 public:
  TensorNWHC() : Tensor() {}
  TensorNWHC(int n, int w, int h, int c)
      : Tensor({n, w, h, c}, Type<float>::datatype) {}
  TensorNWHC(int n, int w, int h, int c,
      const std::vector<float>& ref_vec)
      : Tensor({n, w, h, c}, Type<float>::datatype, ref_vec) {}
  float* data() {
    return Tensor::data<float>();
  }
  float& at(int idx) {
    return Tensor::at<float>(idx);
  }
  std::vector<float> ToVector() const {
    return Tensor::ToVector<float>();
  }

  int GetBatch() const    { return Tensor::GetShape(0); }
  int GetWidth() const    { return Tensor::GetShape(1); }
  int GetHeight() const   { return Tensor::GetShape(2); }
  int GetChannels() const { return Tensor::GetShape(3); }
};
#endif

/**
 * @brief Tensor_<T>
 *
 * @tparam T Cpp data type(float, int8_t, uint8_t, ...)
 *
 * ex) Tensor_<float> tensor(shape_t shape)
 *
 */
template <typename T>
class Tensor_ : public Tensor {
 public:
  Tensor_() : Tensor() {}
  explicit Tensor_(const shape_t& shape) : Tensor(shape, Type<T>::datatype) {}
  Tensor_(const shape_t& shape, const std::vector<T>& ref_vec)
      : Tensor(shape, Type<T>::datatype, ref_vec) {}
};

inline std::ostream& operator<<(std::ostream& os, const DataType& datatype) {
  switch (datatype) {
    case DataType::DT_SINT8:
      os << "SINT8";
      break;
    case DataType::DT_SINT16:
      os << "SINT16";
      break;
    case DataType::DT_UINT8:
      os << "UINT8";
      break;
    case DataType::DT_UINT16:
      os << "UINT16";
      break;
    case DataType::DT_FP32:
      os << "FP32";
      break;
    default:
      os << "not impleted yet";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Tensor::Format& fmat) {
  switch (fmat) {
    case Tensor::Format::NWHC:
      os << "NWHC";
      break;
    case Tensor::Format::NCHW:
      os << "NCHW";
      break;
    case Tensor::Format::NHWC:
      os << "NHWC";
      break;
    default:
      os << "not impleted yet";
  }
  return os;
}
