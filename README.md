# Python Bootcamp Group1
This repository contains the work of Group 1 from the Python Bootcamp in Statistics and Data Analysis for the MSc in High Energy Physics, Astrophysics, and Cosmology at UAB. 


The project focuses on developing a machine learning algorithm to classify galaxy morphologies. 


Pipeline:

1. LoadFilterData.ipynb Loads and Filters the data and creates MergedZooSpecPhotoDR19.csv #EN realitat no funciona, perque triga molt, REVISAR (la resta està fet a partir del arxiu ZooSpecPhotoDR19_torradeflot.csv)
2. SDSS_Pixel_PCA.ipynb Extracts the pixels of the images using parallelized  for a sample of MergedZooSpecPhotoDR19.csv and performs PCA to get the 100 most important pixels columns. Then it merges with MergedZooSpecPhotoDR19.csv and creates PCA_Pixels.csv with the final table
3. Machine Learning algoritm using Random forest 
