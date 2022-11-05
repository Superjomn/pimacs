"lisp.list"() ({
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %res = arith.addi %c1, %c2 : i32
  "lisp.ret" (%res) : (i32) -> (i32)
 }) { symbol="hello" } : () -> (i32)
