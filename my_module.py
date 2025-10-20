import pandas as pd
import numpy as np
from astroquery.utils.tap.core import TapPlus 
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
from PIL import Image as PILImage
from io import BytesIO
import time

# -------------------------------
# Parallelized load of TAP DATA
    
def load_TAP_data_parallel(URL, ra_slices=4, max_workers=4):
    """
    Download data from a TAP service in parallel by splitting the sky into RA slices.

    Parameters
    ----------
    URL : str
        TAP service URL.
    ra_slices : int
        Number of RA partitions (default = 4).
    max_workers : int
        Number of parallel threads (default = 4).

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame of all results.
    """

    # Template for the SQL template for each slice
    # Select all columns from both tables and join them on objid 
    
    adql_template = """
    SELECT
        z.*,
        p.*
    FROM BestDR9.ZooSpec AS z 
    JOIN BestDR7.PhotoObj AS p
    ON p.objid = z.dr7objid
    WHERE {where_clause} 
    """
    # Function that runs an SQL query for one RA slice
    def fetch_slice(ra_min, ra_max):

        where = f"(p.ra >= {ra_min} AND p.ra < {ra_max})" # Defines slice

        adql = adql_template.format(where_clause=where) # Final ADQL for this slice

        try:
            tap = TapPlus(url=URL)          # New connection per thread
            job = tap.launch_job(adql)      # Perform the query
            results = job.get_results()     # Get results (table)
            df = results.to_pandas()
            print(f"Fetched slice RA {ra_min:.1f}–{ra_max:.1f} ({len(df)} rows)")
            return df
        
        except Exception as e:
            print(f"⚠️ Failed slice RA {ra_min:.1f}–{ra_max:.1f}: {e}")
            return pd.DataFrame()
        
    # Create RA slices
    step = 360.0 / ra_slices
    edges = [i * step for i in range(ra_slices)] + [360.0] 

    # Run all slices in parallel
    dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Thread = Working lines
        futures = [executor.submit(fetch_slice, edges[i], edges[i + 1]) for i in range(ra_slices)]  # List of all the pending jobs 
        for f in as_completed(futures):
            df_part = f.result()
            if df_part is not None and not df_part.empty:
                dfs.append(df_part)

    if not dfs:
        print("No data retrieved.")
        return pd.DataFrame()

    df_total = pd.concat(dfs, ignore_index=True)

    # De-duplicate and set index if available
    if "dr7objid" in df_total.columns:
        df_total = df_total.drop_duplicates(subset=["dr7objid"]).set_index("dr7objid")
    else:
        df_total = df_total.drop_duplicates().reset_index(drop=True)

    return df_total


# -------------------------------
# Data cleaning


def clean_data(df):
    """
    Filter and clean the TAP result DataFrame.

    - Converts magnitude and error columns to numeric
    - Removes invalid or extreme values
    - Keeps only confidently classified galaxies
    """

    # --- Convert numeric columns safely ---
    all_numeric = ["modelMag_u", "modelMag_g", "modelMag_r", "modelMag_i", "modelMag_z",
                   "modelMagErr_u", "modelMagErr_g", "modelMagErr_r", "modelMagErr_i", "modelMagErr_z",
                   "p_cs_debiased", "p_el_debiased", "petroR90_r"]
    
    for col in all_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Apply mask ---
    mask = (
        # reasonable magnitude values
        (df["modelMag_u"] > -30)
        & (df["modelMag_g"] > -30)
        & (df["modelMag_r"] > -30)
        & (df["modelMag_i"] > -30)
        & (df["modelMag_z"] > -30)
        &
        # reasonable errors
        (df["modelMagErr_u"] < 0.5)
        & (df["modelMagErr_g"] < 0.05)
        & (df["modelMagErr_r"] < 0.05)
        & (df["modelMagErr_i"] < 0.05)
        & (df["modelMagErr_z"] < 0.1)
        &
        # confident classifications
        ((df["p_cs_debiased"] >= 0.9) | (df["p_el_debiased"] >= 0.9))
        &
        # reasonable galaxy size
        (df["petroR90_r"] * 2 * 1.5 / 0.4 < 64)
        & (df["petroR90_r"] * 2 / 0.4 > 20)
    )

    # --- Columns to keep ---
    cols_to_keep = (
        [
            "specobjid",
            "objid",
            "ra",
            "dec",
            "p_el_debiased",
            "p_cs_debiased",
            "spiral",
            "elliptical",
        ]
        + ["petroR50_r", "petroR90_r"]
        + [f"modelMag_{f}" for f in "ugriz"]
        + [f"extinction_{f}" for f in "ugriz"]
    )

    # --- Apply mask safely ---
    df_filtered = df.loc[mask, cols_to_keep].copy()

    print(f"Cleaned data: {len(df_filtered)} rows remain (from {len(df)} original).")
    return df_filtered



# -------------------------------
# Parallelized SDSS pixel fetch

def fetch_sdss_pixels(df, 
                      image_pixscale=0.4, 
                      image_width_px=64, 
                      image_height_px=64, 
                      max_workers=8,
                      save_path=None):
    """
    Download SDSS cutout images for a DataFrame of objects (RA, DEC, OBJID)
    in parallel and return a flattened pixel DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'ra', 'dec', and 'objid' columns.
    image_pixscale : float, optional
        SDSS scale (arcsec/pixel). Default = 0.4
    image_width_px : int, optional
        Image width in pixels. Default = 64
    image_height_px : int, optional
        Image height in pixels. Default = 64
    max_workers : int, optional
        Number of threads for parallel downloading. Default = 8
    save_path : str, optional
        If given, saves the pixel DataFrame as CSV to this path.

    Returns
    -------
    df_pixels : pandas.DataFrame
        DataFrame with 'objid' and pixel columns (pix_0, pix_1, ..., pix_n).
    """

    URL_TEMPLATE = (
        "https://skyserver.sdss.org/DR19/SkyserverWS/ImgCutout/getjpeg?"
        "ra={ra}&dec={dec}&scale={scale}&width={width}&height={height}"
    )

    # Function to parallelize: fetch one image
    def fetch_image_pixels(row):
        objid = row["objid"]
        ra, dec = row["ra"], row["dec"]
        url = URL_TEMPLATE.format(ra=ra, dec=dec, scale=image_pixscale,
                                  width=image_width_px, height=image_height_px)
        try:
            blob = urllib.request.urlopen(url).read()            # Get the image from the url (via HTTP GET request)
            image = PILImage.open(BytesIO(blob)).convert("L")    # Wrap in a file-like BytesIO buffer, PIL opens the image, converts as grayscale ("L" mode)
            pixels = list(image.getdata())                       # Get pixel values as a flat list
            return objid, pixels
        except Exception as e:
            print(f"⚠️ Error for objid {objid}: {e}")
            return objid, None

    # Parallel download
    pixel_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_image_pixels, row) for idx, row in df.iterrows()] # Schedule each row as a separate task
        for f in as_completed(futures):
            objid, pixels = f.result()
            if pixels is not None:
                pixel_data.append({"objid": objid, **{f"pix_{i}": val for i, val in enumerate(pixels)}}) # Flatten pixels into columns

    df_pixels = pd.DataFrame(pixel_data)

    return df_pixels

# -------------------------------
# SVD on pixel data

def svd_from_pixel_df(df_pixels, id_col='objid', image_width_px=64, image_height_px=64, k=10):
    """
    Perform SVD on flattened pixel data for each image in df_pixels.

    Parameters
    ----------
    df_pixels : pandas.DataFrame
        DataFrame containing 'objid' and pixel columns (pix_0, pix_1, ..., pix_n).
    id_col : str, optional
        Column name for unique object IDs.
    image_width_px : int, optional
        Image width in pixels. Default = 64.
    image_height_px : int, optional
        Image height in pixels. Default = 64.
    k : int, optional
        Number of top singular values to keep.

    Returns
    -------
    svd_df : pandas.DataFrame
        DataFrame with `id_col` and SVD features (svd_comp_1, ..., svd_comp_k).
    """
    if not (0 < k <= min(image_width_px, image_height_px)):
        raise ValueError(f"k must be between 1 and {min(image_width_px, image_height_px)}")

    svd_feature_list = []
    pixel_cols = [c for c in df_pixels.columns if c.startswith("pix_")]

    for idx, row in df_pixels.iterrows():
        raw_id = row[id_col]
        pixel_values = row[pixel_cols].values

        # Reshape flattened pixels back into image matrix
        A = pixel_values.reshape((image_height_px, image_width_px)).astype(np.float64)

        # Perform SVD (we only need singular values)
        sigma = np.linalg.svd(A, compute_uv=False)

        # Keep top-k singular values
        top_k_vals = sigma[:k]
        svd_row = {id_col: raw_id}
        for i in range(k):
            svd_row[f'svd_comp_{i+1}'] = top_k_vals[i]

        svd_feature_list.append(svd_row)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(df_pixels):
            print(f"SVD processed {idx+1}/{len(df_pixels)} images")

    svd_df = pd.DataFrame(svd_feature_list)
    return svd_df


