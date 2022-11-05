func @test_list() {
  "lisp.list"() ({
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %res = arith.addi %c1, %c2 : i32
    "lisp.ret" (%res) : (i32) -> (i32)
   }) { symbol="hello" } : () -> (i32)

  return
}

func @test_constant(%arg0: i32) {
  %1 = "lisp.constant"(){value = 42 : i32} : () -> i32
  return
}

func @test_constant_str(%arg0: i32) {
  %1 = "lisp.constant"(){value = "hello world"} : () -> !lisp.string
  return
}

func @test_call(%arg0: i32) {
  %1 = "lisp.call" (%arg0) { callee = "hello" } : (i32) -> i32
  return
}
