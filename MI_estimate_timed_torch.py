# Mutual Information calculation using GMM-MI-torch, instrumented with timing.
#
# Same structure and CLI as MI_estimate_timed.py, but the estimator is
# gmm-mi-torch (https://github.com/Cosmo-pop/gmm-mi-torch) instead of GMM-MI.
#
# API differences from GMM-MI that shape this file:
#   * There are no param holders. gmm_mi_torch.estimate() takes every
#     hyperparameter as a keyword argument.
#   * estimate() takes ONE array of shape (n_samples, n_variables) rather than
#     separate X and Y, and asserts exactly 2 columns for quantity='MI'. So the
#     latent and category vectors are column_stacked before the call.
#   * It returns (mean, std) where std is None when n_bootstrap=0.
#   * It runs on CPU or GPU and batches internally, which changes how this
#     script should parallelise -- see the --workers note below.
#
# Defaults mirror MI_estimate_timed.py (n_inits=5, n_folds=3, n_bootstrap=100,
# mc_samples=1e5, max_components=50, max_iter=10000) so the two scripts are
# comparable. Two deliberate deviations, both flagged at runtime:
#   * reg_covar defaults to 1e-6, not GMM-MI's 1e-15. The library documents
#     that 1e-15 "would regularise nothing" and is unsafe at float32.
#   * init_type defaults to 'random_responsibilities'; GMM-MI's
#     'random_sklearn' has no counterpart in the port.
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from gmm_mi_torch import estimate
import argparse
import os
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
    description="Mutual Information calculation for a given latent space, "
    "using GMM-MI-torch"
)
parser.add_argument("latent_file", type=str, help="Path to the latent space file")
parser.add_argument("category_file", type=str, help="Path to the category file")
parser.add_argument("file_param", type=str, help="Path to the output file")
parser.add_argument(
    "-reg",
    "--reg_covar",
    type=float,
    default=1e-6,
    help="Regularization covariance. NB the default is 1e-6, the gmm-mi-torch "
    "default, not GMM-MI's 1e-15 which regularises nothing and is unsafe at "
    "float32",
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
# ---- gmm-mi-torch specific options ----
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Torch device to fit and integrate on, e.g. 'cpu' or 'cuda'",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="float64",
    choices=["float64", "float32"],
    help="Working precision. float32 is much faster on GPUs with limited fp64 "
    "throughput (e.g. GB10); float64 is the reference precision",
)
parser.add_argument(
    "--protocol",
    type=str,
    default="batched",
    choices=["batched", "sequential"],
    help="Component-selection protocol. 'batched' fits k=1..max_components at "
    "once (fast on GPU, but pays for all of them unconditionally); "
    "'sequential' is GMM-MI's original protocol and is the like-for-like "
    "comparison against MI_estimate_timed.py",
)
parser.add_argument(
    "--n_bootstrap",
    type=int,
    default=100,
    help="Bootstrap realisations for the errorbar; 0 skips the bootstrap and "
    "yields a NaN std",
)
parser.add_argument(
    "--mc_samples", type=int, default=100000, help="Monte Carlo samples per replicate"
)
parser.add_argument(
    "--max_components",
    type=int,
    default=50,
    help="Largest number of GMM components considered. Mirrors GMM-MI's "
    "default of 50, but note the batched protocol pays for every k, so "
    "lowering this is the single biggest speed knob",
)
parser.add_argument("--n_inits", type=int, default=5, help="Initialisations per fold")
parser.add_argument("--n_folds", type=int, default=3, help="Cross-validation folds")
parser.add_argument("--max_iter", type=int, default=10000, help="Max EM iterations")
parser.add_argument(
    "--init_type",
    type=str,
    default="random_responsibilities",
    choices=["random_responsibilities", "minmax", "kmeans", "randomized_kmeans"],
    help="GMM initialisation scheme",
)
parser.add_argument(
    "--preprocess",
    type=str,
    default="none",
    choices=["none", "gaussianise"],
    help="'gaussianise' applies a per-variable rank-probit transform first. MI "
    "is invariant under it, so it only makes the density estimation easier",
)
parser.add_argument(
    "--em_chunk",
    type=int,
    default=8,
    help="EM replicates per batch. Larger is faster on GPU at higher peak memory",
)
parser.add_argument(
    "--mc_chunk", type=int, default=8, help="Bootstrap replicates per MC integration batch"
)
parser.add_argument(
    "--n_chunk",
    type=int,
    default=None,
    help="If set, stream EM over the sample dimension in chunks of this size to "
    "bound peak memory at large N",
)
parser.add_argument(
    "--workers",
    type=int,
    default=None,
    help="Worker processes for the (cat, lat) sweep. Default: a ProcessPoolExecutor "
    "on CPU, but SERIAL on a non-CPU device, because gmm-mi-torch already "
    "batches internally and one CUDA context per worker wastes memory. "
    "Pass 1 to force serial, or an explicit count to override",
)

args = parser.parse_args()

TORCH_DTYPE = {"float64": torch.float64, "float32": torch.float32}[args.dtype]

# Hyperparameters passed to every estimate() call. Assembled once so the timing
# summary can report exactly what was run.
ESTIMATE_KWARGS = dict(
    protocol=args.protocol,
    n_bootstrap=args.n_bootstrap,
    mc_samples=args.mc_samples,
    max_components=args.max_components,
    n_folds=args.n_folds,
    n_inits=args.n_inits,
    reg_covar=args.reg_covar,
    threshold_fit=args.threshold_fit,
    max_iter=args.max_iter,
    device=args.device,
    dtype=TORCH_DTYPE,
    em_chunk=args.em_chunk,
    mc_chunk=args.mc_chunk,
    init_type=args.init_type,
    preprocess=args.preprocess,
    n_chunk=args.n_chunk,
)

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
print(
    "gmm-mi-torch settings: device:",
    args.device,
    "dtype:",
    args.dtype,
    "protocol:",
    args.protocol,
    "n_bootstrap:",
    args.n_bootstrap,
    "mc_samples:",
    args.mc_samples,
    "max_components:",
    args.max_components,
    "n_inits:",
    args.n_inits,
    "n_folds:",
    args.n_folds,
    "init_type:",
    args.init_type,
    "preprocess:",
    args.preprocess,
    "em_chunk:",
    args.em_chunk,
    "mc_chunk:",
    args.mc_chunk,
    "n_chunk:",
    args.n_chunk,
)
if args.dtype == "float32" and args.reg_covar < 1e-8:
    print(
        f"WARNING: reg_covar={args.reg_covar:g} is below float32 resolution and "
        "will regularise nothing; Cholesky factorisations may fail. Use >=1e-6 "
        "at float32."
    )
file_param = args.file_param
data_dir = "/rds/user/bv322/hpc-work/vae/"

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


def _run_estimate(latent_data, category_data):
    """Call gmm_mi_torch.estimate on one pair and return (mean, std, seconds).

    Unlike GMM-MI's fit_estimate(X=..., Y=...), estimate() takes a single
    (n_samples, 2) array for quantity='MI' -- it asserts the column count.
    """
    pair = np.column_stack(
        (np.asarray(latent_data).ravel(), np.asarray(category_data).ravel())
    )
    # perf_counter is monotonic, so it is immune to wall-clock adjustments.
    t0 = time.perf_counter()
    MI_mean, MI_std = estimate(pair, "MI", **ESTIMATE_KWARGS)
    if args.device.startswith("cuda"):
        # Kernel launches are asynchronous, so without a sync we would time the
        # dispatch rather than the computation.
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    # std is None when n_bootstrap=0; NaN keeps the output array numeric.
    return MI_mean, np.nan if MI_std is None else MI_std, elapsed


def estimate_mi(cat_id, lat_id):
    """Estimate MI for one (category, latent) pair.

    Returns (cat_id, lat_id, MI_mean, MI_std, elapsed), where elapsed is the
    wall-clock time in seconds spent inside estimate(), or None if the pair was
    skipped without doing any work.
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
            MI_mean, MI_std, elapsed = _run_estimate(latent_data, category_data)
            # flush=True: child-process stdout is block-buffered, so without it
            # nothing appears until the worker exits.
            print(
                f"[timing] cat={cat_id} lat={lat_id} estimate took {elapsed:.2f}s",
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
        MI_mean, MI_std, elapsed = _run_estimate(latent_data, category_data)
        print(
            f"[timing] cat={cat_id} lat={lat_id} estimate took {elapsed:.2f}s",
            flush=True,
        )
        return cat_id, lat_id, MI_mean, MI_std, elapsed


def _worker_init():
    """Stop torch threads from oversubscribing the cores across pool workers."""
    torch.set_num_threads(1)


def _resolve_workers():
    """Decide how many processes to fan the (cat, lat) sweep over.

    gmm-mi-torch batches internally and, on GPU, one CUDA context per worker
    wastes memory for no gain -- so a non-CPU device runs serially unless the
    user explicitly overrides.
    """
    if args.workers is not None:
        if args.workers > 1 and args.device != "cpu":
            print(
                f"WARNING: --workers {args.workers} with device={args.device}: "
                "each worker gets its own CUDA context and they contend for the "
                "same GPU. Prefer serial plus larger --em_chunk/--mc_chunk."
            )
        return max(1, args.workers)
    if args.device != "cpu":
        print(
            f"device={args.device}: running the sweep SERIALLY, since "
            "gmm-mi-torch already batches internally. Override with --workers."
        )
        return 1
    return os.cpu_count() or 1


if __name__ == "__main__":
    n_workers = _resolve_workers()
    pairs = [
        (cat_id, lat_id)
        for cat_id in range(len(category_list))
        for lat_id in range(n_latents)
    ]
    timings = []  # (cat_id, lat_id, seconds) for pairs that actually ran

    t_start = time.perf_counter()
    if n_workers == 1:
        results = [estimate_mi(cat_id, lat_id) for cat_id, lat_id in pairs]
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_worker_init
        ) as executor:
            futures = [
                executor.submit(estimate_mi, cat_id, lat_id)
                for cat_id, lat_id in pairs
            ]
            results = [future.result() for future in futures]
    for cat_id, lat_id, MI_mean, MI_std, elapsed in results:
        all_MI_estimates[cat_id, lat_id, 0] = MI_mean
        all_MI_estimates[cat_id, lat_id, 1] = MI_std
        if elapsed is not None:
            timings.append((cat_id, lat_id, elapsed))
    total_wall = time.perf_counter() - t_start

    # Timing summary. On CPU with a pool, the gap between total wall-clock and
    # the sum of per-call times is the parallel speedup actually achieved; run
    # serially for a clean per-call number to compare against GMM-MI.
    print("\n" + "=" * 60)
    print("TIMING SUMMARY -- gmm-mi-torch")
    print("=" * 60)
    print(
        f"device / dtype         : {args.device} / {args.dtype}"
    )
    print(
        f"protocol / max_comp    : {args.protocol} / {args.max_components}"
    )
    print(f"n_bootstrap / mc       : {args.n_bootstrap} / {args.mc_samples}")
    print(f"worker processes       : {n_workers}")
    print(f"total wall-clock       : {total_wall:.1f} s")
    if timings:
        seconds = np.array([t[2] for t in timings])
        print(f"estimate calls         : {len(seconds)}")
        print(f"sum of per-call times  : {seconds.sum():.1f} s")
        print(f"effective speedup      : {seconds.sum() / total_wall:.2f}x")
        print(
            f"mean / median per call : {seconds.mean():.2f} s / "
            f"{np.median(seconds):.2f} s"
        )
        print(
            f"min / max per call     : {seconds.min():.2f} s / "
            f"{seconds.max():.2f} s"
        )
        slowest = max(timings, key=lambda t: t[2])
        print(
            f"slowest pair           : cat={slowest[0]} lat={slowest[1]} "
            f"({slowest[2]:.2f} s)"
        )
        if args.timing_out is not None:
            np.save(args.timing_out, np.array(timings))
            print(f"per-call timings saved to {args.timing_out}")
    else:
        print("no estimate calls were made (all pairs skipped)")
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
        + "/MI_torch_"
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
        + "/MI_torch_"
        + cat_text
        + args.file_param
        + "_reg"
        + str(args.reg_covar)
        + ".npy",
    )
