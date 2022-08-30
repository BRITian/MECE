# MECE
#### a method for enhancing the catalytic efficiency of glycoside hydrolase based on deep neural networks and molecular evolution

---
A lack of effective prediction tools has limited development of high efficiency glycoside hydrolases (GH), which are in high demand for numerous industrial applications. This proof-of-concept study demonstrates the use of a deep neural network and molecular evolution (MECE) platform for predicting catalysis-enhancing mutations in GHs. The MECE platform integrates a deep learning model (DeepGH), trained with 119 GH family protein sequences from the CAZy database. MECE also includes a quantitative mutation design component that uses Gradient-weighted Class Activation Mapping (Grad-CAM) with homologous protein sequences to identify key features for mutation in the target GH, this component can be used in this page.

![image](./plots/a33a78e90d68d865915fb91fa150b2b.jpg)
---


### Requirements:
- python 2 or python 3
- tensorflow == 1.15.0
- keras == 2.3.1
- numpy == 1.19.0
- opencv-python == 3.3.1
- matplotlib == 2.3.5
- scikit-learn == 0.19.2


### USE MECE<br>
*You can use MECE online or download all of the codes to run MECE in local.*
##### Online version:    
[PirD MECE](http://www.elabcaas.cn/pird/mece) 

##### Use in local: <br>
The code MECE.py by the following script in console, the ten-fold models are saved in [./models](./models)<br>
`python MECE.py -data_url <fasta file dir> -data_url <outpot folder dir> `

##### Visualization: <br>
- When you finish run the <mece.py> or get zip file from [PirD MECE](http://www.elabcaas.cn/pird/mece), a csv file will be generated, and also plot the weight in the same dir.<br>
- You can use [plot_logo.r](./plot_logo.r) to plot motif figure or you can use <Chimera - define attribute> to plot 2D structure with weight.<br>
- An example result file for plot motif and 2D sturcture have been saved in [example](./example), the function for generate these files also in [MECE.py](mece.py)<br>
- For plot 2d structure, you must download [UCSF Chimera](https://www.cgl.ucsf.edu/chimera/) or [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/).<br>

##### EXAMPLE:<br>
<img src="./plots/1AYX.png"  style=" height:200px" /><img src="./plots/1AYX_motif.png"  style=" height:200px" />

### Train your own Deep-GH
- Get sequences<br>
About the glycoside hydrolases, of which there are 174 families in the CAZy database, including the unclassified sequences(GH0), and 10 families that contain no reference sequences. For the remaining 164 families, ypu can obtained the corresponding GenBank numbers through the [CAZy website](http://www.cazy.org/) and downloaded the corresponding amino acid sequences using the Batch Entrez port and Biopython toolkit provided by the National Center for Biotechnology Information database ([NCBI](https://www.ncbi.nlm.nih.gov/)).
- Download our dataset<br> 
Or our fasta format dataset are supported in [our website](http://www.elabcaas.cn/pird/mece), you can use [process_dataset.py]("./data/process_dataset.py") and [process_dataset_1.py]("./data/process_dataset_1.py") to convert it to the train/val/test format datasrt.<br>
    1. The [process_dataset.py]("./data/process_dataset.py") is for select GH Family which have more than 10 sequences
    2. the [process_dataset_1.py]("./data/process_dataset_1.py") is for generate 10-fold dataset, split dataset to Train/Val/Test dataset and convert 20 residues to number 1-20.
- Train your own model<br>
The code is [keras_RNN_train_gpu.py](./train_models/keras_RNN_train_gpu.py) in [train_models](./train_models)<br>
    
### Sum weights after Water<br>
- Firstly, you should use the [Water tool](http://emboss.sourceforge.net/apps/release/6.6/emboss/apps/water.html) for perform a pairwise local alignment of each sequence homologous to the wild type.<br>
- Then, based on the results of sequence alignment, the functionally relevant evolutionary feature matrix (Me) of the wild type was obtained by summing all sequence feature matrices using the wild type as the standard.
- Then, The difference between the mutant site Pj,max with the highest importance score in each row of the Me and the wild-type site Pj,wt was compared, and the ploidy relationship (Fj) between Pj,max and Pj,wt was calculated using the division method. 
-Finally, The sites with Fj ≥ 20-fold were selected as single point mutants. The loci with Fj ≥ 20-fold were selected according to the ploidy size to design a multipoint mutant.
- The relevant code is stored in [process_water](./process_water).<br>
    1. [align.py](./process_water/align.py) for align the sequence according to the result of WATER
    2. [Me.py] for(./process_water/Me.py) calculate the value of Me
    3. [Fj.py] for(./process_water/Fj.py) calculate the value of Fj

  
### References
  [MECE: a method for enhancing the catalytic efficiency of glycoside hydrolase based on deep neural networks and molecular evolution]()
  
