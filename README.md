# CNN-SARNet
 Python 2.7.12    
 Tensorflow
 The goal of this project is to provide a way to classify the small image of SAR in ocean, especially two-category.
# Validation Result
 accuracy Top 1 P = 0.9359
# To Run the Project
  main() is in the COSI.py, you can run it to have a train, and you should ensure that other files and COSI.py are in the same folder.
  Command line statement：
  python COSI.py --mode=train --max_steps=50006 --eval_steps=100 --save_steps=2000
  python COSI.py --mode=validation
