func @test_list() {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %res = "lisp.call" (%c1, %c2){callee="+"} : (i32,i32) -> i32
    return
}

func @test_args(%c1:i32, %c2:i32) -> i32 {
    %res = "lisp.call" (%c1, %c2){callee="+"} : (i32,i32) -> i32
    return %res : i32
}
