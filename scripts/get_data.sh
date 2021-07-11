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

function download_coco {
    echo "- Downloading Microsoft COCO Dataset..."
    DATA_DIR=${DATA_BASE}/coco
    mkdir -p ${DATA_DIR}
    pushd ${DATA_DIR}
    wget --continue http://images.cocodataset.org/zips/train2017.zip
    wget --continue http://images.cocodataset.org/zips/val2017.zip
    wget --continue http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    echo "- Unzip data..."
    unzip train2017.zip
    unzip val2017.zip
    unzip annotations_trainval2017.zip
    echo "- Done."
    popd
}

function download_voc {
    echo "- Downloading Pascal VOC Dataset..."
    DATA_DIR=${DATA_BASE}/voc
    mkdir -p ${DATA_DIR}
    pushd ${DATA_DIR}
    wget --continue http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
    wget --continue http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
    wget --continue http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
    wget --continue http://pjreddie.com/media/files/VOC2012test.tar
    echo "- Unzip data..."
    mkdir -p train
    mkdir -p test
    tar -xf VOCtrainval_06-Nov-2007.tar
    mv VOCdevkit/VOC2007 train/
    tar -xf VOCtest_06-Nov-2007.tar
    mv VOCdevkit/VOC2007 test/
    tar -xf VOCtrainval_11-May-2012.tar
    mv VOCdevkit/VOC2012 train/
    tar -xf VOC2012test.tar
    mv VOCdevkit/VOC2012 test
    echo "- Done."
    popd
}

function download_omniglot {
    echo "- Downloading Omniglot Dataset"
    DATA_DIR=${DATA_BASE}/omniglot
    mkdir -p ${DATA_DIR}
    pushd ${DATA_DIR}
    wget --continue https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_background.zip
    wget --continue https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip
    wget --continue https://raw.githubusercontent.com/google-research/meta-dataset/main/meta_dataset/dataset_conversion/dataset_specs/omniglot_dataset_spec.json
    echo "- Unzip data..."
    mkdir -p images
    unzip -q images_background.zip
    unzip -q images_evaluation.zip
    mv images_background/* images/
    mv images_evaluation/* images/
    rm -rf images_background/
    rm -rf images_evaluation/
    rm images_background.zip
    rm images_evaluation.zip
    popd
}

for d_name in $@; do
    echo "Handling $d_name";
    download_${d_name};
done
