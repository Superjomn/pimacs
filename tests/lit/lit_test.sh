#!/bin/bash
set -ex

function test {
    filename=$1
    ./parser_entry.py $filename --sema 1 | filecheck $filename
}

test ./basic_test.pim
test ./org-element.pim
# TODO: support optional type and enable this
#test ./buffer.pim

./parser_entry.py --sema 1 name-binding.pim --mark-unresolved 1 | filecheck name-binding.pim
./parser_entry.py --sema 1 class-template.pim --mark-unresolved 1 | filecheck ./class-template.pim
