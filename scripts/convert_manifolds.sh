#!/bin/bash
#
# convert_manifolds.sh: This script performs the automated conversion of
# triangulations of 2-manifolds and 3-manifolds (originally described in
# mixed lexicographic format) to compressed JSON files.
#
# Usage:
#
#   $ sh scripts/convert_manifolds.sh
#
# This script assumes it runs in the base directory of the package. 

STELLAR_URL=https://zenodo.org/api/records/11474260/files-archive

# Make sure that we bail out directly in case any of the commands below
# fail for whatever reason.
set -e

curl $STELLAR_URL --output data/manifolds.zip
unzip -d data/ data/manifolds.zip

cat data/2_manifolds_all.txt      data/2_manifolds_10_all.txt      > data/2_manifolds.txt
cat data/2_manifolds_all_type.txt data/2_manifolds_10_all_type.txt > data/2_manifolds_type.txt
cat data/2_manifolds_all_hom.txt  data/2_manifolds_10_all_hom.txt  > data/2_manifolds_hom.txt

echo "Converting 2-manifolds..."

python -m mantra.lex_to_json            data/2_manifolds.txt      \
                             --type     data/2_manifolds_type.txt \
                             --homology data/2_manifolds_hom.txt  \
          > 2_manifolds.json

gzip --best 2_manifolds.json

echo "Converting 3-manifolds..."

cat data/3_manifolds_all.txt      data/3_manifolds_10_all.txt      data/manifolds_lex_d3_deg5.txt      > data/3_manifolds.txt
cat data/3_manifolds_all_type.txt data/3_manifolds_10_all_type.txt data/manifolds_lex_d3_deg5_type.txt > data/3_manifolds_type.txt
cat data/3_manifolds_all_hom.txt  data/3_manifolds_10_all_hom.txt  data/manifolds_lex_d3_deg5_hom.txt  > data/3_manifolds_hom.txt

python -m mantra.lex_to_json            data/3_manifolds.txt      \
                             --type     data/3_manifolds_type.txt \
                             --homology data/3_manifolds_hom.txt  \
          > 3_manifolds.json

gzip --best 3_manifolds.json
