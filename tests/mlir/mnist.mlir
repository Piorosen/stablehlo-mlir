module @jit__unnamed_wrapped_function_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x28x28x1xf32>) -> (tensor<1x10xf32> {jax.result_info = ""}) {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c = stablehlo.constant dense<4> : tensor<i32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x1x32xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<32xf32>
    %cst_3 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x32x64xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_5 = stablehlo.constant dense_resource<__elided__> : tensor<3136x256xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<256xf32>
    %cst_7 = stablehlo.constant dense_resource<__elided__> : tensor<256x10xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
    %0 = stablehlo.convolution(%arg0, %cst_1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x1xf32>, tensor<3x3x1x32xf32>) -> tensor<1x28x28x32xf32>
    %1 = stablehlo.reshape %cst_2 : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x28x28x32xf32>
    %4 = call @relu(%3) : (tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<f32>
    %6 = "stablehlo.reduce_window"(%4, %5) <{window_dimensions = array<i64: 1, 2, 2, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %39 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %39 : tensor<f32>
    }) : (tensor<1x28x28x32xf32>, tensor<f32>) -> tensor<1x14x14x32xf32>
    %7 = stablehlo.convert %c : (tensor<i32>) -> tensor<f32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<1x14x14x32xf32>
    %9 = stablehlo.divide %6, %8 : tensor<1x14x14x32xf32>
    %10 = stablehlo.convolution(%9, %cst_3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x32xf32>, tensor<3x3x32x64xf32>) -> tensor<1x14x14x64xf32>
    %11 = stablehlo.reshape %cst_4 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %13 = stablehlo.add %10, %12 : tensor<1x14x14x64xf32>
    %14 = call @relu_0(%13) : (tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
    %15 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<f32>
    %16 = "stablehlo.reduce_window"(%14, %15) <{window_dimensions = array<i64: 1, 2, 2, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %39 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %39 : tensor<f32>
    }) : (tensor<1x14x14x64xf32>, tensor<f32>) -> tensor<1x7x7x64xf32>
    %17 = stablehlo.convert %c : (tensor<i32>) -> tensor<f32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<1x7x7x64xf32>
    %19 = stablehlo.divide %16, %18 : tensor<1x7x7x64xf32>
    %20 = stablehlo.reshape %19 : (tensor<1x7x7x64xf32>) -> tensor<1x3136xf32>
    %21 = stablehlo.dot_general %20, %cst_5, contracting_dims = [1] x [0] : (tensor<1x3136xf32>, tensor<3136x256xf32>) -> tensor<1x256xf32>
    %22 = stablehlo.reshape %cst_6 : (tensor<256xf32>) -> tensor<1x256xf32>
    %23 = stablehlo.add %21, %22 : tensor<1x256xf32>
    %24 = call @relu_1(%23) : (tensor<1x256xf32>) -> tensor<1x256xf32>
    %25 = stablehlo.dot_general %24, %cst_7, contracting_dims = [1] x [0] : (tensor<1x256xf32>, tensor<256x10xf32>) -> tensor<1x10xf32>
    %26 = stablehlo.reshape %cst_8 : (tensor<10xf32>) -> tensor<1x10xf32>
    %27 = stablehlo.add %25, %26 : tensor<1x10xf32>
    %28 = stablehlo.reduce(%27 init: %cst) applies stablehlo.maximum across dimensions = [1] : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
    %29 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %30 = stablehlo.maximum %29, %28 : tensor<1xf32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %32 = stablehlo.broadcast_in_dim %31, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x10xf32>
    %33 = stablehlo.subtract %27, %32 : tensor<1x10xf32>
    %34 = stablehlo.exponential %33 : tensor<1x10xf32>
    %35 = stablehlo.reduce(%34 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x10xf32>
    %38 = stablehlo.divide %34, %37 : tensor<1x10xf32>
    return %38 : tensor<1x10xf32>
  }
  func.func private @relu(%arg0: tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28x32xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x28x28x32xf32>
    return %1 : tensor<1x28x28x32xf32>
  }
  func.func private @relu_0(%arg0: tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14x64xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x14x14x64xf32>
    return %1 : tensor<1x14x14x64xf32>
  }
  func.func private @relu_1(%arg0: tensor<1x256xf32>) -> tensor<1x256xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x256xf32>
    return %1 : tensor<1x256xf32>
  }
}
