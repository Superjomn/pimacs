;; my-functions-test.el
(require 'ert)

(ert-deftest test-square ()
  "Test the square function."
  (should (= (+ 0 1) 1)))

;; Run all tests
(ert-run-tests-batch-and-exit)
