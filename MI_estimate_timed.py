# Mutual Information calculation, instrumented with timing of the EstimateMI calls.
#
# Identical to MI_estimate.py except that every fit_estimate() call is timed
# inside its worker process, the duration is returned alongside the MI values,
# and a summary (total wall-clock vs. sum of per-call times) is printed at the end.
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from gmm_mi.mi import EstimateMI
from gmm_mi.param_holders import (
    GMMFitParamHolder,
    SelectComponentsParamHolder,
    MIDistParamHolder,
)
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

plt.rc("text", usetex=False)
plt.rcParams["axes.formatter.min_exponent"] = 4
mpl.rcParams["agg.path.chunksize"] = 10000
plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.right"] = True
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["figure.autolayout"] = True
plt.rcParams["savefig.bbox"] = "tight"

parser = argparse.ArgumentParser(
    description="Mutual Information calculation for a given latent space"
)
parser.add_argument("latent_file", type=str, help="Path to the latent space file")
parser.add_argument("category_file", type=str, help="Path to the category file")
parser.add_argument("file_param", type=str, help="Path to the output file")
parser.add_argument(
    "-reg", "--reg_covar", type=float, default=1e-15, help="Regularization covariance"
)
parser.add_argument(
    "-thresh", "--threshold_fit", type=float, default=1e-5, help="Threshold fit"
)
parser.add_argument("-lat", "--latent_MI", action="store_true", help="Latent-latent MI")
parser.add_argument(
    "-lat_param",
    "--lat_param",
    action="store_true",
    help="Mutual Information between one latent and a parametrization of it",
)
parser.add_argument(
    "-bin_spec",
    "--bin_spectra",
    action="store_true",
    help="Bin spectra into n bins for MI calculation",
)
parser.add_argument(
    "-zmin", "--zmin", type=float, default=None, help="Minimum redshift for selection"
)
parser.add_argument(
    "-zmax", "--zmax", type=float, default=None, help="Maximum redshift for selection"
)
parser.add_argument(
    "--lim_cat",
    default=None,
    help="category on which to limit the dataset for the MI calculation",
)
parser.add_argument(
    "-lcmin", "--lim_cat_min", type=float, help="Minimum value for category selection"
)
parser.add_argument(
    "-lcmax",
    "--lim_cat_max",
    type=float,
    default=None,
    help="Maximum value for category selection",
)
parser.add_argument(
    "-timing_out",
    "--timing_out",
    type=str,
    default=None,
    help="Optional path to save the per-call timings as a .npy array of "
    "(cat_id, lat_id, seconds) rows",
)

args = parser.parse_args()

print(
    "latent file: ",
    args.latent_file,
    " category file: ",
    args.category_file,
    " file param: ",
    args.file_param,
    " reg_covar: ",
    args.reg_covar,
    " threshold_fit: ",
    args.threshold_fit,
    "latent_MI: ",
    args.latent_MI,
    "lat_param: ",
    args.lat_param,
    "bin_spectra: ",
    args.bin_spectra,
    "zmin: ",
    args.zmin,
    "zmax: ",
    args.zmax,
    "lim_cat: ",
    args.lim_cat,
    "lim_cat_min: ",
    args.lim_cat_min,
    "lim_cat_max: ",
    args.lim_cat_max,
)
file_param = args.file_param
data_dir = "/rds/user/bv322/hpc-work/vae/"

# parameters for every GMM fit that is being run
gmm_fit_params = GMMFitParamHolder(
    threshold_fit=args.threshold_fit,
    reg_covar=args.reg_covar,
)
select_components_params = SelectComponentsParamHolder(n_inits=5, n_folds=3)
mi_dist_params = MIDistParamHolder(n_bootstrap=100, MC_samples=1e5)

# Reading in the data
if args.latent_file.endswith(".npy"):
    latents = np.load(args.latent_file)
else:
    latents = pd.read_csv(args.latent_file, sep=",").values
if args.category_file.endswith(".npy"):
    categories = np.load(args.category_file)
elif args.category_file.endswith(".csv"):
    categories = pd.read_csv(args.category_file)
elif args.category_file.endswith(".npz"):
    with np.load(args.category_file) as cat_data:
        categories = cat_data["arr_0"][0:100000, :]
else:
    categories = latents
# Whether we need to restrict the categories to a validation set (vs latent-latent MI)
if "spectra" in args.category_file:
    val_index = np.load(data_dir + file_param + "/validation_set" + file_param + ".npy")
    cat_text = "lat_spectra_unnormalized_"
    blue_lim = 277
    red_lim = 4500
    categories = categories[val_index, blue_lim:red_lim]  # this is already unnormalized
    if args.bin_spectra:  # bin spectra into n number of binscate
        n_bins = 12  # in khedarlarian they use 12 between 3400-7000 A
        bin_edges = np.linspace(3400, 7000, 13)
        # Initialize an array to hold binned data: (N_samples, 12)
        binned_categories = np.zeros((categories.shape[0], n_bins))
        for i in range(12):
            low, high = bin_edges[i], bin_edges[i + 1]
            # Average flux density in wavelength units
            wavelength = np.load("wavelength_array.npy")
            wavelength = wavelength[blue_lim:red_lim]
            idx = (wavelength >= low) & (wavelength < high)
            if np.any(idx):
                binned_categories[:, i] = np.mean(categories[:, idx], axis=1)
            else:
                binned_categories[:, i] = 0.0
        categories = binned_categories
        cat_text += f"binned{n_bins}_"
    print("categories shape", categories.shape)
elif args.category_file.startswith("catalog") or "mock" in args.category_file:
    print("using the validation set")
    val_index = np.load(
        data_dir + file_param + "/validation_set" + args.file_param + ".npy"
    )
    cat_text = "lat_prop"
    categories = categories.iloc[val_index]
    print("categories shape", categories.shape)
    print("categories columns", categories.columns)
else:
    cat_text = "latent_latent_"
category_list = [f"category_{i}" for i in range(categories.shape[1])]
if "full_val" in args.latent_file:
    latents = latents[:, -1]
if args.lat_param:  # select only one latent and a parametrization of it
    properties = pd.read_csv(
        "/rds/user/bv322/hpc-work/mock_catalog_Ch1_26_mlim_short_dn4k.csv"
    )
    properties = properties[val_index]
    latents = latents[:, 1]
    latent_param = (
        0.504 * properties["log10Mstar"] + 0.159 * properties["log10SFR"] - 5.899
    )
    latents = np.column_stack((latents, latent_param))
# Make a redshift selection for the categories and associated latents
if args.zmin is not None and args.zmax is not None:
    if "z" in categories.columns:
        z = categories["z"].values
        mask = (z >= args.zmin) & (z <= args.zmax)
        categories = categories[mask]
        latents = latents[mask]
    else:
        print(
            "No redshift column found in categories for redshift selection. No redshift selection applied."
        )
n_latents = latents.shape[1]
print("categories shape", categories.shape, "latents shape", latents.shape)

# Make a category selection for the categories and associated latents
if args.lim_cat is not None and args.lim_cat_min is not None:
    cat_id = categories.columns.get_loc(args.lim_cat)
    if cat_id < categories.shape[1]:
        cat_data = categories.iloc[:, cat_id].values
        if args.lim_cat_min is None:
            args.lim_cat_min = np.min(cat_data)
        if args.lim_cat_max is None:
            args.lim_cat_max = np.max(cat_data)
        mask = (cat_data >= args.lim_cat_min) & (cat_data <= args.lim_cat_max)
        categories = categories[mask]
        latents = latents[mask]
    else:
        print(
            "Category index for selection is out of bounds. No category selection applied."
        )
    print(
        "category for selection: ",
        categories.columns[cat_id],
        "categories shape after selection",
        categories.shape,
        "latents shape",
        latents.shape,
    )
# changing categories and latent size to make it quicker
if "spectra" in args.category_file:
    categories = categories[:3000, :]
    latents = latents[:3000, :]

all_MI_estimates = np.zeros((len(category_list), n_latents, 2))


def estimate_mi(cat_id, lat_id):
    """Estimate MI for one (category, latent) pair.

    Returns (cat_id, lat_id, MI_mean, MI_std, elapsed), where elapsed is the
    wall-clock time in seconds spent inside fit_estimate(), or None if the pair
    was skipped without doing any work.
    """
    if args.latent_MI:
        if lat_id != cat_id:
            print(
                "Calculating MI for latent: ",
                cat_id,
                " and latent dimension: ",
                lat_id,
            )
            category_data = categories[:, cat_id]
            latent_data = latents[:, lat_id]
            mi_estimator = EstimateMI(
                gmm_fit_params=gmm_fit_params,
                select_components_params=select_components_params,
            )
            # perf_counter is monotonic, so it is immune to wall-clock adjustments.
            t0 = time.perf_counter()
            MI_mean, MI_std = mi_estimator.fit_estimate(
                X=latent_data,
                Y=category_data,
                mi_dist_params=mi_dist_params,
            )
            elapsed = time.perf_counter() - t0
            # flush=True: child-process stdout is block-buffered, so without it
            # nothing appears until the worker exits.
            print(
                f"[timing] cat={cat_id} lat={lat_id} fit_estimate took {elapsed:.2f}s",
                flush=True,
            )
            return cat_id, lat_id, MI_mean, MI_std, elapsed
        else:
            print(
                "Skipping MI for latent dimension: ",
                lat_id,
                " and category: ",
                cat_id,
            )
            return cat_id, lat_id, 1, 1, None
    else:
        if isinstance(categories, pd.DataFrame):
            category_list = categories.columns.tolist()
            category_data = categories[category_list[cat_id]].values
        else:
            category_list = [f"wavelength_{i}" for i in range(categories.shape[1])]
            category_data = categories[:, cat_id]
        print(
            "Calculating MI for category: ",
            category_list[cat_id],
            " and latent dimension: ",
            lat_id,
            "categories type",
            type(categories),
            "latents type",
            type(latents),
        )
        latent_data = latents[:, lat_id]
        mi_estimator = EstimateMI(
            gmm_fit_params=gmm_fit_params,
            select_components_params=select_components_params,
        )
        t0 = time.perf_counter()
        MI_mean, MI_std = mi_estimator.fit_estimate(
            X=latent_data,
            Y=category_data,
            mi_dist_params=mi_dist_params,
        )
        elapsed = time.perf_counter() - t0
        print(
            f"[timing] cat={cat_id} lat={lat_id} fit_estimate took {elapsed:.2f}s",
            flush=True,
        )
        return cat_id, lat_id, MI_mean, MI_std, elapsed


if __name__ == "__main__":
    timings = []  # (cat_id, lat_id, seconds) for pairs that actually ran
    t_start = time.perf_counter()
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(estimate_mi, cat_id, lat_id)
            for cat_id in range(len(category_list))
            for lat_id in range(n_latents)
        ]
        for future in futures:
            cat_id, lat_id, MI_mean, MI_std, elapsed = future.result()
            all_MI_estimates[cat_id, lat_id, 0] = MI_mean
            all_MI_estimates[cat_id, lat_id, 1] = MI_std
            if elapsed is not None:
                timings.append((cat_id, lat_id, elapsed))
    total_wall = time.perf_counter() - t_start

    # Timing summary. The gap between total wall-clock and the sum of per-call
    # times is the parallel speedup actually achieved by the process pool.
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"total wall-clock       : {total_wall:.1f} s")
    if timings:
        seconds = np.array([t[2] for t in timings])
        print(f"fit_estimate calls     : {len(seconds)}")
        print(f"sum of per-call times  : {seconds.sum():.1f} s")
        print(f"effective speedup      : {seconds.sum() / total_wall:.2f}x")
        print(f"mean / median per call : {seconds.mean():.2f} s / "
              f"{np.median(seconds):.2f} s")
        print(f"min / max per call     : {seconds.min():.2f} s / "
              f"{seconds.max():.2f} s")
        slowest = max(timings, key=lambda t: t[2])
        print(
            f"slowest pair           : cat={slowest[0]} lat={slowest[1]} "
            f"({slowest[2]:.2f} s)"
        )
        if args.timing_out is not None:
            np.save(args.timing_out, np.array(timings))
            print(f"per-call timings saved to {args.timing_out}")
    else:
        print("no fit_estimate calls were made (all pairs skipped)")
    print("=" * 60 + "\n")

    if args.zmin is not None and args.zmax is not None:
        cat_text += f"z{args.zmin:.2f}-{args.zmax:.2f}_"
    if args.lim_cat is not None and args.lim_cat_min is not None:
        cat_text += f"cat{args.lim_cat}_min{args.lim_cat_min:.2f}"
    if args.lim_cat_max is not None:
        cat_text += f"_max{args.lim_cat_max:.2f}"
    np.save(
        data_dir
        + file_param
        + "/MI_"
        + cat_text
        + args.file_param
        + "_reg"
        + str(args.reg_covar)
        + ".npy",
        all_MI_estimates,
    )
    print("saved with file_param", file_param)
    print(
        "MI estimates saved in",
        data_dir
        + file_param
        + "/MI_"
        + cat_text
        + args.file_param
        + "_reg"
        + str(args.reg_covar)
        + ".npy",
    )
