# find . -name "*.png" -exec convert {} `basename {} .png`.pdf \;
find . -name "*.png" -exec convert {} {}.pdf \;
rename 's/\.png\.pdf/\.pdf/' *.pdf
