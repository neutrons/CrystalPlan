#!/bin/bash
#Script to test ubmatrixreader.py

echo
echo "------ Sample from Gruber (1973) ------"
python ./ubmatrixreader.py sampleubMatrix3.txt -v

echo
echo "------ Sample from Krivy, Gruber (1976) ------"
python ./ubmatrixreader.py sampleubMatrix2.txt -v

echo
echo "------ Sample from Christina ------"
python ./ubmatrixreader.py sampleubMatrix.txt -v

