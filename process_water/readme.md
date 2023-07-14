In the first step, you need to use the WATER tool to make a two by two comparison
step_2_after_water.py is used to align the target sequence obtained using water with the results of a two-by-two comparison of homologous sequences.
In the third step, you need to use the Grad-CAM method to calculate the feature matrix (Mi) of homologous sequence
step_4_after_water.py is the fourth step, which averages all sequence feature matrices according to the sequence comparison results, using the wild type as the standard, to obtain the functionally relevant evolutionary feature matrix (Me) for the wild type.

