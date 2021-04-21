## **Package Usage**

#### **Install Matlab, python engines and Toolbox**  
``` cd "MATLAB-ROOT/extern/engines/python" ```  
``` python setup.py install  ```

> Statistics and Machine Learning Toolbox  
> Image Processing Toolbox


#### **Install bse_ml package**  
``` pip install bse_ml ```  


#### **Skull Striping using DACN**
Put '.nii.gz' file into a folder then run the command in this folder  
```bse -i input [optional settings]```  

Required settings:  
-i \<input filename>  

Optional settings:  
-o \<output filename>  

Example for 5444HD.nii.gz:    
``` bse_ml -i 5444HD.nii.gz ```  
 or   
``` bse_ml -i 5444HD.nii.gz  -o 5444HD_output.nii.gz```  

The results will be saved
