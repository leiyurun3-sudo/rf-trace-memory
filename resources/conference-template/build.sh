#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass1.stdout
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass2.stdout
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass3.stdout
printf '%s\n' 'Built build/main.pdf in three passes.'
