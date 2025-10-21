# Python Bootcamp Group 1 — Galaxy Morphology Classification

This repository contains the work of Group 1 from the Python Bootcamp in Statistics and Data Analysis for the MSc in High Energy Physics, Astrophysics, and Cosmology at the Autonomous University of Barcelona.

The goal of this project is to classify galaxy morphologies (spiral vs elliptical) using data from the Sloan Digital Sky Survey (SDSS) and the Galaxy Zoo project.

## Project Goals 

- Load and merge astronomical data from TAP (Table Access Protocol) services.

<<<<<<< HEAD
1. LoadFilterData.ipynb Loads and Filters the data and creates the file 'MergedZooSpecPhotoDR19.csv'. 
    -The information loaded ads up to 138.960 galaxies and after the filtering reduces to 13.460. 
2. SDSS_Pixel_PCA.ipynb Extracts the pixels of the images of a sample of 'MergedZooSpecPhotoDR19.csv' using the coordinates (ra,dec). 
    -Download SDSS cutout images for a sample of 'MergedZooSpecPhotoDR19.csv' of objects (RA, DEC, OBJID) in parallel and return a flattened pixel DataFrame.
    -The procedure is parallelized for faster results. Then SVD is applied to get the k, (k=10), most important singular values which describe the structure of the image. 
    -Then it merges with MergedZooSpecPhotoDR19.csv and creates SVD_Pixels.csv with the final table. 
3. Machine Learning algoritm using Random forest, Logistic Regression, and SVM.
    -The target variable is constructed to be 0 for elliptical galaxies and 1 for spiral galaxies.
    -We note an imbalance between the classes, with class 1 being approximately 3.5x larger than class 0.
    -Nonetheless, the results show that the algorithms were able to find an effective linear separation of the data points.
    -For application purposes, we would choose the logistic regression model since it is linear in its decision boundary and easy to calibrate.
  
=======
- Clean and filter the catalog to remove invalid or uncertain measurements.

- Download and preprocess galaxy images (SDSS cutouts) in parallel.

- Extract key features from images using Singular Value Decomposition (SVD) to reduce the dimensionality of the dataset.
>>>>>>> 80f061e27235ecbe69365b58c294f3dcd09b9085

- Train a machine learning model to classify galaxies based on combined catalog and image features.

<<<<<<< HEAD
IDEAS README: 
- objectius
- project structure
- ML algorithms (1 or many?)
=======
## Project Structure

`├── mymodule.py                 # Core functions for data loading, cleaning, image fetching, and SVD`

`├── LoadFilterData.ipynb        # Loads and filters TAP data, creates FilteredZooSpecPhotoDR19.csv`

`├── SDSS_Pixel_PCA.ipynb        # Downloads SDSS images, applies SVD, creates SVD_Pixels.csv`

`├── Final_ML_Code.ipynb         # Machine Learning code for galaxy morphology. Uses SVM, RF, LogReg

`└── README.md                   # This file` 

## Functional Overview (mymodule.py)
1. `load_TAP_data_parallel()`

- Parallelized data download from a TAP service by dividing the sky into RA slices.

- Connects to a TAP service (e.g., SDSS or Galaxy Zoo) using `astroquery.tap`.

- Fetches data slices in parallel using `ThreadPoolExecutor`.

- Joins and deduplicates the results into a single `pandas.DataFrame`.

`df = load_TAP_data_parallel(URL="https://some.tap.service/tap", ra_slices=4, max_workers=4)`

- **Returns:** combined DataFrame of all results.

2. `clean_data()`

- Cleans and filters the catalog data.

- Converts relevant columns to numeric types.

- Removes outliers and invalid magnitude/error values.

- Keeps only confidently classified galaxies (based on `p_cs_debiased` or `p_el_debiased`).

`df_clean = clean_data(df)`

**Returns:** filtered DataFrame with valid galaxy entries.

3. `fetch_sdss_pixels()`

- Downloads SDSS cutout images for a given catalog of galaxies in parallel.

- Retrieves images using ra and dec coordinates.

- Converts each image to grayscale and flattens it into a pixel vector.

`df_pixels = fetch_sdss_pixels(df_clean, max_workers=8)`

**Returns:** pixel DataFrame with columns like objid, pix_0, pix_1, ..., pix_n.

4. `svd_from_pixel_df()`

- Performs Singular Value Decomposition (SVD) on image pixel data.

- Reshapes each flattened image into a matrix.

- Computes SVD and extracts the top k singular values.

- Produces a compact representation of image structure.

`svd_df = svd_from_pixel_df(df_pixels, k=10)`

**Returns:** DataFrame of SVD features (`svd_comp_1`, `svd_comp_2`, …).


## Data Processing Pipeline
### Step 1: Load & Filter Data

**Notebook:** `LoadFilterData.ipynb`

- Loads SDSS + Galaxy Zoo data from TAP with `load_TAP_data_parallel()`.

- Filters it with `clean_data()`.

- Saves the merged catalog as `FilteredZooSpecPhotoDR19.csv`.

- From an initial 159 099 galaxies, the clean subset reduces to 15 582.

### Step 2: Extract Image Pixels & Features

**Notebook:** `SDSS_Pixel_SVD.ipynb`

- Downloads SDSS cutout images using `fetch_sdss_pixels()`.

- Reduces pixel data using SVD for dimensionality reduction.

- Merges results with the filtered catalog.

- Outputs `SVD_Pixels.csv` with combined catalog + image features.

### Step 3: Machine Learning

**Notebook:** `Final_ML_code.ipynb`
- Defines target and training variables (X, y)

- Defines train, validation, and test splits (60%, 20%, 20%)

- Trains each of the 3 candidate models on the train split

- Uses the validation split to select the best hyperparameters from a given set

- Finally, compares the performance of each of the optimized models on the test set. Each of the models performs relatively equal, aroun 100% accuracy

- Conclusion: We would choose the LogReg model due to its linear decision boundary, ease of interpretation for feature weights, and simple calibration
>>>>>>> 80f061e27235ecbe69365b58c294f3dcd09b9085
