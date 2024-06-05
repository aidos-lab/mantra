#!/bin/sh
#
# compare_triangulations.sh: Debug utility script to compare two sets of
# triangulations. The diff is performed using the `diff` utility, so the
# ideal output of this script is *empty* except for some status lines.
#
# Usage:
#   
#   $ sh/scripts/compare_triangulations.sh 2_manifolds.json.gz 3_manifolds.json.gz
#
# This script *only* works on compressed files.

TMP_ONE=$(mktemp)
TMP_TWO=$(mktemp)

for FIELD in "triangulation" "name"; do
  echo -n Field "$FIELD"...
  

  gunzip -c $1 | jq ".[] | {$FIELD}" | jq -c > $TMP_ONE
  gunzip -c $2 | jq ".[] | {$FIELD}" | jq -c > $TMP_TWO

  DIFF=$(diff -q $TMP_ONE $TMP_TWO)

  if [ "$DIFF" ]; then
    echo differs!
  else
    echo coincides.
  fi
done

rm $TMP_ONE
rm $TMP_TWO
