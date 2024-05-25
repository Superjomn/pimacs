#!/bin/bash
set -ex

function test {
    filename=$1
    ./parser_entry.py $filename --sema 1 | filecheck $filename
}

test ./basic_test.pis
test ./org-element.pis
test ./buffer.pis

./parser_entry.py  --sema 1 name-binding.pis --mark-unresolved 1 | filecheck name-binding.pis
