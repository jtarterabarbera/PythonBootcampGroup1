import pandas as pd
from astroquery.utils.tap.core import TapPlus  # Library to query astronomical databases, TAP services
import urllib.request # Library to download data and images from URLs
from PIL import Image as PILImage # For image processing
from io import BytesIO # To handle binary data in memory as if it were a file
from concurrent.futures import ThreadPoolExecutor, as_completed # For parallel tasks
from sklearn.preprocessing import StandardScaler # Data normalization
from sklearn.decomposition import PCA # Dimensionality reduction via Principal Component Analysis (PCA)

def safe_to_numeric(col): # To convert each column of a Data Frame to numeric system
    try:
        return pd.to_numeric(col, errors="raise")
    except Exception: # Leave the column unchanged if it can't be fully coverted
        return col  

def load_TAP_data(URL):
    # Connect to the TAP service
    tap = TapPlus(url=URL)
    # To obtain all the data
    adql = """
    SELECT TOP 100000
        z.*,
        p.*
    FROM BestDR9.ZooSpec AS z 
    JOIN BestDR7.PhotoObj AS p
    ON p.objid = z.dr7objid
    """

    # Run query
    job = tap.launch_job(adql)
    results = job.get_results()
    df = results.to_pandas()  # Convert to pandas DataFrame

    # Convert columns to numeric and set an index
    df = df.apply(safe_to_numeric).set_index("dr7objid")

    return df

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
        # Delete unreasonable errors
        (df["modelMagErr_u"] < 0.5)
        & (df["modelMagErr_g"] < 0.05)
        & (df["modelMagErr_r"] < 0.05)
        & (df["modelMagErr_i"] < 0.05)
        & (df["modelMagErr_z"] < 0.1)
        &
        # High classification certainty
        ((df["p_cs_debiased"] >= 0.9) | (df["p_el_debiased"] >= 0.9))
        &
        # Medium size
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

    # Function to fetch one image
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


def apply_pca_to_pixels(df_pixels, df_filtered, n_components=100, save_path=None):
    """
    Apply PCA to pixel data and merge the resulting components with metadata.

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

    # To merge PCA results with filtered metadata
    df_final = pd.merge(df_filtered, df_pca, on="objid", how="inner")

    # Saving
    if save_path:
        df_final.to_csv(save_path, index=False)

    return df_final

