
#pragma once

#include <cassert>
#include <cstddef>

namespace rnn {
namespace cuda {

struct CuMatrix {
  unsigned rows;
  unsigned cols;

  // The pitch of the rows of the weights matrix in bytes.
  size_t pitch;

  // Data pointers allocated with cudaMallocPitch. Logical size is (rows * cols).
  // Elements are stored in row major format (inner loop traverses across a given row).
  float *data;

  CuMatrix() = default;
  CuMatrix(unsigned rows, unsigned cols, size_t pitch, float *data)
      : rows(rows), cols(cols), pitch(pitch), data(data) {}

  static inline CuMatrix FromBuffer(const CuMatrix &buf, unsigned blockRows, unsigned index) {
    return CuMatrix(blockRows, buf.cols, buf.pitch,
                    (float *)((char *)buf.data + index * blockRows * buf.pitch));
  }
};

struct TargetOutput {
  unsigned batchSize; // equal to the number of rows in the matrix actually used.

  // each sample is a row vector in the targetOutput matrix.
  CuMatrix value;

  TargetOutput() = default;
  TargetOutput(unsigned batchSize, CuMatrix value) : batchSize(batchSize), value(value) {
    assert(batchSize > 0);
    assert(batchSize <= value.rows);
  }
};

struct ConnectionActivation {
  unsigned batchSize; // equal to the number of rows in the matrix actually used.

  CuMatrix activation;
  CuMatrix derivative;

  ConnectionActivation() = default;
  ConnectionActivation(unsigned batchSize, CuMatrix activation, CuMatrix derivative)
      : batchSize(batchSize), activation(activation), derivative(derivative) {
    assert(batchSize > 0);
    assert(batchSize <= activation.rows);
    assert(activation.rows == derivative.rows && activation.cols == derivative.cols);
  }
};

struct LayerBatchDeltas {
  unsigned batchSize; // equal to the number of rows in the matrix actually used.
  CuMatrix delta;

  LayerBatchDeltas() = default;
  LayerBatchDeltas(unsigned batchSize, CuMatrix delta) : batchSize(batchSize), delta(delta) {
    assert(batchSize > 0);
    assert(batchSize <= delta.rows);
  }
};
}
}
