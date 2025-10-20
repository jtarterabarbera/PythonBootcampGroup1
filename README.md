# Python Bootcamp Group 1 — Galaxy Morphology Classification

This repository contains the work of Group 1 from the Python Bootcamp in Statistics and Data Analysis for the MSc in High Energy Physics, Astrophysics, and Cosmology at the Autonomous University of Barcelona.

The goal of this project is to classify galaxy morphologies (e.g., spiral vs elliptical) using data from the Sloan Digital Sky Survey (SDSS) and the Galaxy Zoo project.

### Project Goals 

Load and merge astronomical data from TAP (Table Access Protocol) services.

Clean and filter the catalog to remove invalid or uncertain measurements.

Download and preprocess galaxy images (SDSS cutouts) in parallel.

Extract key features from images using Singular Value Decomposition (SVD) or PCA.

Train a machine learning model to classify galaxies based on combined catalog and image features.

### Project Structure
├── mymodule.py                 # Core functions for data loading, cleaning, image fetching, and SVD
├── LoadFilterData.ipynb        # Loads and filters TAP data, creates MergedZooSpecPhotoDR19.csv
├── SDSS_Pixel_PCA.ipynb        # Downloads SDSS images, applies SVD, creates SVD_Pixels.csv
├── Final_ML_Code.ipynb         # Machine Learning code for galaxy morphology
└── README.md                   # This file

### Functional Overview (mymodule.py)
1. load_TAP_data_parallel()

Parallelized data download from a TAP service by dividing the sky into RA slices.

Connects to a TAP service (e.g., SDSS or Galaxy Zoo) using astroquery.tap.

Fetches data slices in parallel using ThreadPoolExecutor.

Joins and deduplicates the results into a single pandas.DataFrame.

df = load_TAP_data_parallel(URL="https://some.tap.service/tap", ra_slices=4, max_workers=4)


Returns: combined DataFrame of all results.

2. clean_data()

Cleans and filters the catalog data.

Converts relevant columns to numeric types.

Removes outliers and invalid magnitude/error values.

Keeps only confidently classified galaxies (based on p_cs_debiased or p_el_debiased).

df_clean = clean_data(df)


Returns: filtered DataFrame with valid galaxy entries.

3. fetch_sdss_pixels()

Downloads SDSS cutout images for a given catalog of galaxies in parallel.

Retrieves images using ra and dec coordinates.

Converts each image to grayscale and flattens it into a pixel vector.

Optionally saves the result to a CSV file.

df_pixels = fetch_sdss_pixels(df_clean, max_workers=8)


Returns: pixel DataFrame with columns like objid, pix_0, pix_1, ..., pix_n.

4. svd_from_pixel_df()

Performs Singular Value Decomposition (SVD) on image pixel data.

Reshapes each flattened image into a matrix.

Computes SVD and extracts the top k singular values.

Produces a compact representation of image structure.

svd_df = svd_from_pixel_df(df_pixels, k=10)


Returns: DataFrame of SVD features (svd_comp_1, svd_comp_2, …).

### Data Processing Pipeline
Step 1: Load & Filter Data

Notebook: LoadFilterData.ipynb

Loads SDSS + Galaxy Zoo data from TAP.

Filters it with clean_data().

Saves the merged catalog as MergedZooSpecPhotoDR19.csv.

From an initial 138,960 galaxies, the clean subset reduces to 13,460.

Step 2: Extract Image Pixels & Features

Notebook: SDSS_Pixel_PCA.ipynb

Downloads SDSS cutout images using fetch_sdss_pixels().

Reduces pixel data using PCA or SVD for dimensionality reduction.

Merges results with the filtered catalog.

Outputs PCA_Pixels.csv with combined catalog + image features.

Step 3: Machine Learning

Notebook: MachineLearning.ipynb

Applies a Random Forest classifier to predict galaxy morphology.

Uses both catalog and image-based features.

Evaluates model performance using standard ML metrics.