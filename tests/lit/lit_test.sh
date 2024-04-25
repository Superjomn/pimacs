#!/bin/bash
set -ex

function test {
    filename=$1
    ./parser_entry.py --filename $filename | filecheck $filename
}

test ./basic_test.pis
