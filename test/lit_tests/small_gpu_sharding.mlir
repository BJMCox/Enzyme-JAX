// RUN: enzymexlamlir-opt %s --split-input-file --lower-small-program-control --verify-diagnostics | FileCheck %s

module {
  sdy.mesh @empty_mesh = <[]>
  func.func @main(%key: tensor<2xui32>) -> tensor<2xui32> {
    %result = sdy.sharding_constraint %key <@empty_mesh, [{}]> : tensor<2xui32>
    return %result : tensor<2xui32>
  }
}

// CHECK: func.func @main(%[[KEY:.*]]: tensor<2xui32>) -> tensor<2xui32>
// CHECK-NEXT: return %[[KEY]] : tensor<2xui32>

// -----

module {
  sdy.mesh @mesh = <["x"=2]>
  func.func @main(%key: tensor<2xui32>) -> tensor<2xui32> {
    // expected-error @+1 {{small GPU profile supports only empty-mesh sharding constraints}}
    %result = sdy.sharding_constraint %key <@mesh, [{"x"}]> : tensor<2xui32>
    return %result : tensor<2xui32>
  }
}
