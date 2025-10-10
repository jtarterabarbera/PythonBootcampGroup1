import pandas as pd
from astroquery.utils.tap.core import TapPlus  # library to query astronomical databases, TAP services
import urllib.request
from PIL import Image as PILImage
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def safe_to_numeric(col, **kwargs):
    try:
        return pd.to_numeric(col, errors="raise")
    except Exception:
        return col


def load_TAP_data(URL, ra_slices: int = 8, max_workers: int = 4):
    """Download all rows from the TAP service by splitting
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
    def fetch_slice(ra_min: float, ra_max: float) -> pd.DataFrame:
        where = f"(p.ra >= {ra_min}) AND (p.ra < {ra_max})"
        adql = adql_template.format(where_clause=where)
        try:
            tap_local = TapPlus(url=URL)
            # maxrec=-1 requests all available rows for this slice
            job = tap_local.launch_job(adql, maxrec=-1)
            results = job.get_results()
            return results.to_pandas()
        except Exception as e:
            print(f"⚠️ RA slice [{ra_min}, {ra_max}) failed: {e}")
            return pd.DataFrame()

    # Build RA slices across [0, 360)
    ra_slices = max(1, int(ra_slices))
    step = 360.0 / ra_slices
    edges = [i * step for i in range(ra_slices)] + [360.0]

    dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(ra_slices):
            ra_min = edges[i]
            ra_max = edges[i + 1]
            futures.append(executor.submit(fetch_slice, ra_min, ra_max))

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

from concurrent.futures import ThreadPoolExecutor
from astroquery.utils.tap.core import TapPlus
import pandas as pd

def load_TAP_data_simpler(URL, ra_slices=8, max_workers=4):
    """
    Descarrega totes les dades del TAP dividint el cel en franges de RA
    i fent les consultes en paral·lel.
    """
    adql_template = """
    SELECT z.*, p.*
    FROM BestDR9.ZooSpec AS z 
    JOIN BestDR7.PhotoObj AS p ON p.objid = z.dr7objid
    WHERE p.ra >= {ra_min} AND p.ra < {ra_max}
    """

    def fetch_slice(ra_min, ra_max):
        try:
            tap = TapPlus(url=URL)
            query = adql_template.format(ra_min=ra_min, ra_max=ra_max)
            job = tap.launch_job(query, maxrec=-1)
            return job.get_results().to_pandas()
        except Exception as e:
            print(f"Error al rang RA [{ra_min}, {ra_max}): {e}")
            return pd.DataFrame()

    step = 360 / ra_slices
    ranges = [(i * step, (i + 1) * step) for i in range(ra_slices)]

    dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for df_part in ex.map(lambda r: fetch_slice(*r), ranges):
            if not df_part.empty:
                dfs.append(df_part)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Conversió numèrica i índex
    if "dr7objid" in df.columns:
        df = df.apply(safe_to_numeric, errors="ignore")
        df = df.drop_duplicates(subset="dr7objid").set_index("dr7objid")

    return df



def clean_data(df):
    # Apply the mask to filter the DataFrame
    mask = (
        # correct magnitudes
        (df["modelMag_u"] > -30)
        & (df["modelMag_g"] > -30)
        & (df["modelMag_r"] > -30)
        & (df["modelMag_i"] > -30)
        & (df["modelMag_z"] > -30)
        # reasonable errors
        & (df["modelMagErr_u"] < 0.5)
        & (df["modelMagErr_g"] < 0.05)
        & (df["modelMagErr_r"] < 0.05)
        & (df["modelMagErr_i"] < 0.05)
        & (df["modelMagErr_z"] < 0.1)
        # very certain about the classification
        & ((df["p_cs_debiased"] >= 0.9) | (df["p_el_debiased"] >= 0.9))
        # medium size
        & (df["petroR90_r"] * 2 * 1.5 / 0.4 < 64)
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
# -------------------------------
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

    # -------------------------------
    # Parallel download
    # -------------------------------
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

    # 1️⃣ Normalize pixel data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pixel_data)

    # 2️⃣ Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # 3️⃣ Build a PCA DataFrame
    df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    df_pca.insert(0, "objid", obj_ids)

    # 4️⃣ Merge PCA results with filtered metadata
    df_final = pd.merge(df_filtered, df_pca, on="objid", how="inner")

    # 5️⃣ Optionally save
    if save_path:
        df_final.to_csv(save_path, index=False)

    return df_final

