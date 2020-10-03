#!/bin/bash

mkdir -p data
mkdir -p data/input
mkdir -p data/output
mkdir -p logs
mkdir -p pickle

# download digit-recognizer
cd ./data/input/
kaggle competitions download -c digit-recognizer -p .
unzip ./*.zip
rm -rf *.zip