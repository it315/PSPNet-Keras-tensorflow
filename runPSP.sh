#!/bin/bash

for i in {1..53}
do
    python3.5 pspnet.py -m pspnet50_ade20k -i /home/ripsuser1/PSPNet-Keras-tensorflow-Iris/inputimages/31July2/$i.png -o /home/ripsuser1/PSPNet-Keras-tensorflow-Iris/outputimages/31July2/$i.jpg
   
done

echo OK
