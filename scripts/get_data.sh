#!/bin/bash

DATA_BASE=${DATA_BASE:-"${HOME}/awnas/data"}

function download_ptb {
    echo "- Downloading Penn Treebank (PTB)"
    mkdir -p ${DATA_BASE}/ptb
    pushd ${DATA_BASE}/ptb
    wget --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    tar -xzf simple-examples.tgz

    cp ./simple-examples/data/ptb.train.txt train.txt
    cp ./simple-examples/data/ptb.test.txt test.txt
    cp ./simple-examples/data/ptb.valid.txt valid.txt
    popd
}

function download_tiny-imagenet {
    echo "- Downloading Tiny-Imagenet"
    DATA_DIR=${DATA_BASE}/tiny-imagenet
    mkdir -p ${DATA_DIR}
    pushd ${DATA_DIR}
    wget --continue http://cs231n.stanford.edu/tiny-imagenet-200.zip
    unzip tiny-imagenet-200.zip

    echo "soft linking image files..."
    # train subdir by class
    ln -s tiny-imagenet-200/train train
    # val subdir by class
    cat tiny-imagenet-200/val/val_annotations.txt \
	| awk -v DATA_DIR=${DATA_DIR} '{system("mkdir -p val/" $2); system("ln -s " DATA_DIR "/tiny-imagenet-200/val/images/" $1 " val/" $2)}'
    popd
}

for d_name in $@; do
    echo "Handling $d_name";
    download_${d_name};
done
