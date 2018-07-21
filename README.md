= Generate Savefiles from pretrained ckpt file wiht TF slim =

To generate metagraph, index and data to use pre-trained model from google, 
with ckechpoint.

Alpha release

1. Clone https://github.com/tensorflow/models to use TF slim & model
(in case of this project, the model is inception v3) 
2. Download ckpt file from TFslim 
Go https://github.com/tensorflow/models/tree/master/research/slim
Scroll down to Readme and Get your tgz file.
3. copy included files & unextract tgz into models/research/slim/ 
4. run python script

The script doing...
* generate model using TF slim library
* load old ckpt file
* Inference 1 sample image to confirm the weights are normally restored.
I use inception_v3_2016_08_28.tar.gz and get results  like below
> Probability 67.65% => [Pembroke, Pembroke Welsh corgi]
Probability 22.00% => [Cardigan, Cardigan Welsh corgi]
Probability 0.11% => [dingo, warrigal, warragal, Canis dingo]
Probability 0.09% => [nipple]
Probability 0.07% => [Ibizan hound, Ibizan Podenco]

* save model 
you can see these fiies
inception_v3.data-00000-of-00001
inception_v3.index
inception_v3.meta

after then you can use that files like below

saver = tf.train.import_meta_graph("inception_v3.meta")
saver.restore(sess, "inception")



