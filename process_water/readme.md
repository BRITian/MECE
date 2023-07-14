- In the first step, you need to use the WATER tool to make a two by two comparison
  
- [step_2_after_water.py](step_2_after_water.py)  is the second step, used to align the target sequence obtained using water with the results of a two-by-two comparison of homologous sequences.

- In the third step, you need to use the [Grad-CAM method](../MECE.py) to calculate the feature matrix (Mi) of homologous sequence

- [step_4_calculation_Me_matrix.py](step_4_calculation_Me_matrix.py) is the fourth step, which averages all sequence feature matrices according to the sequence comparison results, using the wild type as the standard, to obtain the functionally relevant evolutionary feature matrix (Me) for the wild type.

- [step_5_calculation_fj.py](step_5_calculation_fj.py) is the fifth step to caculate Fj, it was defined as the multiple of the mutation site Pj,max with the highest importance score in each row of the matrix Me to the wild-type site Pj,wt ，with higher scores suggesting greater probability of improved catalytic efficiency.

- You can try this process with the sequence <b>c1754</b> from our experiment: [example_file](c1754.fa).
