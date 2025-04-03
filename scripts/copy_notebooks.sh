#! /bin/bash

# The script copies the notebooks in the examples folder to the notebooks folder 
# in the documentation. Each notebook is converted to html and added to sphinx. 

# Usage 
#
#  $ ./scripts/copy_notebooks.sh
#

# Make sure that exits directly with an error in case any of the commands fails. 
set -e

# Clean the target directory.
rm -f ./docs/source/notebooks/*.ipynb

# Copy all notebooks in the examples folder to the docs.
cp ./examples/*.ipynb ./docs/source/notebooks