# CNN-SARNet
 Python 2.7.12    
 Tensorflow    
 The goal of this project is to provide a way to classify the small image of SAR in ocean, especially two-category.    

## Validation Result
 accuracy Top 1 P = 0.9359
## To Run the Project
  main() is in the `COSI.py`, you can run it to have a train, and you should ensure that `other files` and `COSI.py` are in the same folder.                
  Command line statement：    
  ```
  python COSI.py --mode=train --max_steps=50006 --eval_steps=100 --save_steps=2000
  python COSI.py --mode=validation
  ……
  ```
## project structure
   `COSI.py`       ：includes the main functions and FLAGS    
   `cnnnet.py`     ：includes the structure of the CNN-SARNet    
   `DI.py`         ：data read, preprocessing and enhancement    
   `log.py`       ：bulid the logger files of the project        
   `train.py`     ：algorithm of training network    
   `validation.py`：validation method    
   `inference.py` ：inference for single test images

## Inquiring
   If you had some questions, you would email to me : [wyx_wx@foxmail.com]                
   If you are also committed to applying `Deep Learning` to remote sensing (especially SAR ) research, you could also contact me for academic exchange.
