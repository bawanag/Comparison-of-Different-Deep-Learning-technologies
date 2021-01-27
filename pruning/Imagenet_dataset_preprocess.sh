#trainset 大文件解压之后是1k个小的分类文件，得依次解压
dir=/nobackup/sc19yt/project/pytorch_model_zoo_pretrain_model/pruning_finetune_imagenet/train/
for x in `ls *.tar`
do
	filename=`basename $x .tar`
	mkdir $filename
	tar -xvf $x -C ./$filename
	rm x
done