#!/bin/bash
set -ex

function test {
    filename=$1
    ./parser_entry.py $filename --sema 1 | filecheck $filename
}

test ./basic_test.pis
test ./org-element.pis
# TODO: support optional type and enable this
#test ./buffer.pis

./parser_entry.py --sema 1 name-binding.pis --mark-unresolved 1 | filecheck name-binding.pis
./parser_entry.py --sema 1 class-template.pis --mark-unresolved 1 | filecheck ./class-template.pis
