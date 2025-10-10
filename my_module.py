import pandas as pd
from astroquery.utils.tap.core import TapPlus 
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
from PIL import Image as PILImage
from io import BytesIO
import time

# -------------------------------
# Parallelized load of TAP DATA
    
def safe_to_numeric(col):
    """
    Convert a pandas Series to numeric, ignoring errors.
    If conversion fails, returns the original Series.
    """

    return pd.to_numeric(col, errors='ignore') 


def load_TAP_data(URL, ra_slices: int = 8, max_workers: int = 4):
    """
    Download all rows from the TAP service (no TOP limit) by splitting
    the sky in RA slices and querying them in parallel.

    Parameters
    ----------
    URL : str
        TAP service URL.
    ra_slices : int, optional
        Number of RA partitions in [0, 360) to query in parallel. Default = 8.
    max_workers : int, optional
        Number of threads for parallel downloading. Default = 4.

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame of all results with index set to 'dr7objid'.
    """

    # Base ADQL without TOP cap; we add a WHERE clause per RA slice
    adql_template = """ 
    SELECT
        z.*,
        p.*
    FROM BestDR9.ZooSpec AS z 
    JOIN BestDR7.PhotoObj AS p
    ON p.objid = z.dr7objid
    WHERE {where_clause}
    """

    # Helper to fetch a single RA slice. Use a fresh TapPlus per thread to be safe.
    def fetch_slice(ra_min: float, ra_max: float, wrap: bool = False) -> pd.DataFrame:
        where = (
            f"(p.ra >= {ra_min}) OR (p.ra < {ra_max})" if wrap
            else f"(p.ra >= {ra_min}) AND (p.ra < {ra_max})"
        )
        adql = adql_template.format(where_clause=where)
        try:
            tap_local = TapPlus(url=URL)
            job = tap_local.launch_job(adql)
            results = job.get_results()
            return results.to_pandas()
        except Exception as e:
            print(f"⚠️ RA slice [{ra_min}, {ra_max}) failed: {e}")
            return pd.DataFrame()

    # Build RA slices across [0, 360)
    ra_slices = max(1, int(ra_slices))
    edges = [i * (360.0 / ra_slices) for i in range(ra_slices)]

    dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(ra_slices):
            ra_min = edges[i]
            ra_max = edges[(i + 1) % ra_slices]
            wrap = i == ra_slices - 1  # last slice wraps around to 0
            futures.append(executor.submit(fetch_slice, ra_min, ra_max, wrap))

        for f in as_completed(futures):
            df_part = f.result()
            if df_part is not None and not df_part.empty:
                dfs.append(df_part)

    if not dfs:
        return pd.DataFrame()

    # Concatenate and clean
    df = pd.concat(dfs, ignore_index=True)

    # Convert columns safely to numeric and set index
    df = df.apply(safe_to_numeric)

    # Remove any accidental duplicates on join key
    if "dr7objid" in df.columns:
        df = df.drop_duplicates(subset=["dr7objid"]).set_index("dr7objid")
    else:
        # Fallback: set a generic index
        df = df.drop_duplicates().reset_index(drop=True)

    return df



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
            print(f"✅ Fetched slice RA {ra_min:.1f}–{ra_max:.1f} ({len(df)} rows)")
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


def load_TAP_data_parallel_simple(URL, ra_slices=8, max_workers=4):
    """
    Download TAP data in parallel using RA slices, assuming every slice has data.
    """
    # SQL template for each slice
    adql_template = """
    SELECT
        z.*,
        p.*
    FROM BestDR9.ZooSpec AS z 
    JOIN BestDR7.PhotoObj AS p
    ON p.objid = z.dr7objid
    WHERE p.ra >= {ra_min} AND p.ra < {ra_max}
    """

    # Function to parallelize: fetch one RA slice
    def fetch_slice(ra_min, ra_max):

        adql = adql_template.format(ra_min=ra_min, ra_max=ra_max)

        tap = TapPlus(url=URL)          # New connection per thread
        job = tap.launch_job(adql)      # Perform the query
        results = job.get_results()     # Get results (table)
        df = results.to_pandas()

        return df

    # RA slice boundaries
    slice_size = 360.0 / ra_slices
    edges = [i * slice_size for i in range(ra_slices + 1)]  # include end point

    
    # Run all slices in parallel
    dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Thread = Working lines
        futures = [executor.submit(fetch_slice, edges[i], edges[i + 1]) for i in range(ra_slices)] # Schedule one task per slice; returns Future objects (start running as workers are free)
        for f in as_completed(futures): # As each job completes
            df_part = f.result()
            dfs.append(df_part)

    
    df_total = pd.concat(dfs, ignore_index=True)

    # Remove duplicates on 'dr7objid'
    if 'dr7objid' in df_total.columns:
        df_total = df_total.drop_duplicates(subset=['dr7objid']).set_index('dr7objid')

    return df_total

# -------------------------------
# Data cleaning

def clean_data(df):
    # Apply the mask to filter the DataFrame
    mask = (
        # Correct magnitudes
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
        # very certain about the classification
        ((df["p_cs_debiased"] >= 0.9) | (df["p_el_debiased"] >= 0.9))
        &
        # medium size
        (df["petroR90_r"] * 2 * 1.5 / 0.4 < 64)
        & (df["petroR90_r"] * 2 / 0.4 > 20)
    )

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

    df_filtered = df[mask][cols_to_keep]
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
            blob = urllib.request.urlopen(url).read()
            image = PILImage.open(BytesIO(blob)).convert("L")
            pixels = list(image.getdata())
            return objid, pixels
        except Exception as e:
            print(f"⚠️ Error for objid {objid}: {e}")
            return objid, None

    # Parallel download
    pixel_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_image_pixels, row) for idx, row in df.iterrows()]
        for f in as_completed(futures):
            objid, pixels = f.result()
            if pixels is not None:
                pixel_data.append({"objid": objid, **{f"pix_{i}": val for i, val in enumerate(pixels)}})

    df_pixels = pd.DataFrame(pixel_data)

    return df_pixels



def fetch_sdss_pixels_version2(df, 
                      image_pixscale=0.4, 
                      image_width_px=64, 
                      image_height_px=64, 
                      max_workers=4,
                      save_path=None,
                      retries=3, # number of retries per image, avoid timeouts
                      timeout=5): # timeout per image in seconds
    """
    Download SDSS cutout images for a DataFrame of objects (RA, DEC, OBJID)
    in parallel, with retries and timeout, and return a flattened pixel DataFrame.
    """

    URL_TEMPLATE = (
        "https://skyserver.sdss.org/DR19/SkyserverWS/ImgCutout/getjpeg?"
        "ra={ra}&dec={dec}&scale={scale}&width={width}&height={height}"
    )
    # Function to parallelize: fetch one image with retries
    def fetch_image_pixels(row):
        """Fetch pixels for one object, with retries and timeout."""
        objid, ra, dec = row["objid"], row["ra"], row["dec"]

        for attempt in range(1, retries + 1):
            try:
                url = URL_TEMPLATE.format(ra=ra, dec=dec, scale=image_pixscale,
                                          width=image_width_px, height=image_height_px)
                
                blob = urllib.request.urlopen(url, timeout=timeout).read() # Get the image from the url (via HTTP GET request)
                bytes = blob.read()                                        # Gives the raw binary data (the JPEG file contents).
                image = PILImage.open(BytesIO(bytes)).convert("L")         # Wrap in a file-like BytesIO buffer, PIL opens the image, converts as grayscale ("L" mode)
                pixels = list(image.getdata())                             # Get pixel values as a flat list

                return objid, pixels
            
            except Exception as e:
                print(f"⚠️ objid {objid}, attempt {attempt}/{retries} failed: {e}")
                time.sleep(0.5)  # short delay before retrying

        # All retries failed
        return objid, None

    # Parallel download
    pixel_data = [] # List of dicts with objid and pixel columns
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_image_pixels, row) for _, row in df.iterrows()] # Schedule each row as a separate task
        for f in as_completed(futures):
            objid, pixels = f.result()
            if pixels is not None:
                pixel_data.append({"objid": objid, **{f"pix_{i}": val for i, val in enumerate(pixels)}}) # Flatten pixels into columns

    # Combine all into a DataFrame
    df_pixels = pd.DataFrame(pixel_data)

    # Optionally save to CSV
    if save_path is not None:
        df_pixels.to_csv(save_path, index=False)

    return df_pixels



# -------------------------------
# PCA on pixel data


def apply_pca_to_pixels(df_pixels, n_components=100, save_path=None):
    """
    Apply PCA to pixel data.

    Parameters
    ----------
    df_pixels : pandas.DataFrame
        DataFrame containing 'objid' and pixel columns (pix_0, pix_1, ...)
    df_filtered : pandas.DataFrame
        Original metadata DataFrame containing 'objid' and other columns
    n_components : int, optional
        Number of PCA components to keep. Default = 100
    save_path : str, optional
        If provided, saves the resulting merged DataFrame to CSV.

    Returns
    -------
    df_final : pandas.DataFrame
        Metadata DataFrame merged with PCA components, ready for ML.
    """

    # Separate objid from pixel data
    obj_ids = df_pixels["objid"]
    pixel_data = df_pixels.drop(columns=["objid"])

    # To normalize pixel data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pixel_data)

    # For applying the PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # For building a PCA DataFrame
    df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    df_pca.insert(0, "objid", obj_ids)

    return df_pca

