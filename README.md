# deep-sign-vision

All of the code was made in python 3.6
The only dependancies are Tensorflow,mathplotlib,and numpy
###dataset
The dataset issued from the university of exter can be found here : http://kahlan.eps.surrey.ac.uk/FingerSpellingKinect2011/fingerspelling5.tar.bz2

The dataset consist of 5 directory for each person making the sign, each with 24 directory, one for each sign, named with the letter it represent.

For use by the neural network, merge the 5 directory and only keep the 24 directory with the name of the letter that the sign represent. The name of the files themselves does not matter.

###Preparation
To launch the preparation of the dataset, use `datasetPrep.py`with the name of the input directory( that contains the 24 sub directory), and the name of the output directory (in wich the 24 directory with the new files will be created):
`python datasetPrep.py -d [dir] -o [outputDir] -a prep`

###Finetuning
To launch the finetuning, just use `python finetune.py [pathToPrepDir]`
