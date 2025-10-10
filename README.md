# Python Bootcamp Group 1
This repository contains the work of Group 1 from the Python Bootcamp in Statistics and Data Analysis for the MSc in High Energy Physics, Astrophysics, and Cosmology at Autonomous University of Barcelona. 


The project focuses on developing a machine learning algorithm to classify galaxy morphologies. 

The project combines catalog data (photometric and morphological parameters) with image data (pixel-level information from SDSS) to train and evaluate the classifier.

Pipeline:

1. LoadFilterData.ipynb Loads and Filters the data and creates the file 'MergedZooSpecPhotoDR19.csv'. The information loaded ads up to 138.960 galaxies and after the filtering reduces to 13.460. 
2. SDSS_Pixel_PCA.ipynb Extracts the pixels of the images of a sample of 'MergedZooSpecPhotoDR19.csv' using the coordinates (ra,dec). Download SDSS cutout images for a sample of 'MergedZooSpecPhotoDR19.csv' of objects (RA, DEC, OBJID) in parallel and return a flattened pixel DataFrame. The procedure is parallelized for faster results. Then to reduce the size of information a PCA is applied to get the 100 most important pixels columns. Then it merges with MergedZooSpecPhotoDR19.csv and creates PCA_Pixels.csv with the final table. 
3. Machine Learning algoritm using Random forest 




IDEAS README: 
- objectius
- project structure
