# ReHyDIL

We provide a simple demo for testing the code.

In the demo, incremental learning can be done using the following commands:  
python demo.py --img_mode t1  
python demo.py --img_mode t2 --prev_img_mode t1  
python demo.py --img_mode flair --prev_img_mode t2-t1  
python demo.py --img_mode t1ce --prev_img_mode flair-t2-t1  

The BraTS19 dataset needs to be used, and the preprocessing method is as described in the paper.

The complete code and documentation will be released after acceptance.
