#!/usr/bin/env python3

# Takes ONE PARAMETER:
# Example: if called with parameter jiagu,
# This script will replace all families THAT ARE NOT jiagu
# to NOT_jiagu



import sys

FAMILY = sys.argv[1]

NEGATIVE_LABEL = "NOT_" + FAMILY


for line in sys.stdin.readlines():
    (sha, label) = line.strip().split(',')
    if label != FAMILY:
        label = NEGATIVE_LABEL
    print(','.join([sha, label]))
