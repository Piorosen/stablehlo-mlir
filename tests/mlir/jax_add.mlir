func.func @main(%arg0: tensor<1xf32>, %bias: tensor<1xf32>) -> tensor<1xf32> {
  %0 = stablehlo.add %arg0, %bias: tensor<1xf32>
  return %0 : tensor<1xf32>
}
