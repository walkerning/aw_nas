echo "- Downloading Penn Treebank (PTB)"
mkdir -p data/ptb
pushd data/ptb
wget --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xzf simple-examples.tgz

cp ./simple-examples/data/ptb.train.txt train.txt
cp ./simple-examples/data/ptb.test.txt test.txt
cp ./simple-examples/data/ptb.valid.txt valid.txt
popd
