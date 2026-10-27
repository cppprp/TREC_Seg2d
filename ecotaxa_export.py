"""
ecotaxa_export.py
-----------------
Pull instance segmentations from WebKnossos, turn each 3D instance into 2D
"specimen-like" projections, and package / upload them as an EcoTaxa project.

Per instance:
  1. Find its bounding box (one coarse scan of the segmentation layer).
  2. Read raw + labels for that box from WebKnossos, at the finest mag that
     keeps the crop under --max_side voxels.
  3. PCA of the mask voxels gives the body axes (long, mid, short); the masked
     raw crop is resampled into that frame so the long axis is horizontal.
  4. Project along the short axis ("lateral", rank 0 — the largest silhouette,
     like an animal lying flat on a ZooScan tray) and along the mid axis
     ("dorsal", rank 1).
       sum : line integral of attenuation — a simulated radiograph, closest to
             transmitted-light instruments (ZooScan). Default.
       max : maximum intensity projection — shows only the densest structures.
  5. Normalise with one window for the whole volume (so grey levels are
     comparable between objects), invert to dark-on-white, optionally resample
     to --pixel_um, and save as PNG.

Output: <output>/<sample>/ with images/ + ecotaxa_<sample>.tsv, zipped to
<output>/<sample>.zip. With --upload, the zip is uploaded to EcoTaxa and an
import job is started into an existing (--project_id) or new (--new_project)
project. Classification itself is then done in the EcoTaxa web UI (Predict /
manual sorting); object_wk_label maps every object back to its 3D instance.

Usage:
    export WK_TOKEN=...                       # WebKnossos API token
    export ECOTAXA_USER=... ECOTAXA_PASSWORD=...
    python ecotaxa_export.py \\
        --sample   POR_10to40_20231023_AM_01_epo_02 \\
        --output   /scratch/asvetlove/TREC/ecotaxa \\
        --upload --project_id 12345
"""

import argparse
import csv
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.transform import rescale
from tqdm import tqdm


# ---------------------------------------------------------------------------
# WebKnossos
# ---------------------------------------------------------------------------

def open_wk_dataset(name: str, url: str, token: str, raw_layer: str, seg_layer: str):
    import webknossos as wk
    with wk.webknossos_context(url=url, token=token):
        ds = wk.Dataset.open_remote(name)
    return ds, ds.get_layer(raw_layer), ds.get_layer(seg_layer)


_thread_local = threading.local()


def _thread_layers(*open_args):
    """
    One RemoteDataset per worker thread: a dataset re-enters its own (stateful)
    webknossos_context on access, which is not safe to share between threads.
    """
    if not hasattr(_thread_local, 'layers'):
        _thread_local.layers = open_wk_dataset(*open_args)[1:]
    return _thread_local.layers


def _available_mags(layer) -> list:
    return sorted(layer.mags.keys(), key=lambda m: m.max_dim)


def _pick_mag(layer, factor: int):
    """Finest available mag that is at least `factor` (or the coarsest one)."""
    mags = _available_mags(layer)
    for m in mags:
        if m.max_dim >= factor:
            return m
    return mags[-1]


def scan_instances(seg_layer, scan_mag: int = 4, slab: int = 128) -> dict:
    """
    One pass over the segmentation layer at a coarse mag, slab by slab along z.
    Returns {label: dict(lo, hi, voxels)} with lo/hi in Mag(1) xyz coordinates
    (hi exclusive) and an estimated Mag(1) voxel count.
    """
    import webknossos as wk

    mag = _pick_mag(seg_layer, scan_mag)
    f = mag.max_dim
    view = seg_layer.get_mag(mag)
    full = seg_layer.bounding_box.align_with_mag(mag, ceil=True)
    x0, y0, z0 = full.topleft.to_tuple()
    sx, sy, sz = full.size.to_tuple()
    print(f"Scanning segmentation at mag {f} (bbox {full.topleft.to_tuple()} + {full.size.to_tuple()})")

    found = {}
    step = slab * f
    for zs in tqdm(range(z0, z0 + sz, step), desc='scan', unit='slab'):
        dz = min(step, z0 + sz - zs)
        box = wk.BoundingBox((x0, y0, zs), (sx, sy, dz))
        arr = view.read(absolute_bounding_box=box)[0]            # (x, y, z) at mag f
        if not arr.any():
            continue
        counts = np.bincount(arr.ravel())
        for idx, sl in enumerate(ndimage.find_objects(arr)):
            if sl is None:
                continue
            label = idx + 1
            lo = np.array([x0, y0, zs]) + np.array([s.start for s in sl]) * f
            hi = np.array([x0, y0, zs]) + np.array([s.stop for s in sl]) * f
            n = int(counts[label]) * f ** 3
            if label in found:
                e = found[label]
                e['lo'] = np.minimum(e['lo'], lo)
                e['hi'] = np.maximum(e['hi'], hi)
                e['voxels'] += n
            else:
                found[label] = dict(lo=lo, hi=hi, voxels=n)

    # A coarse mag can shave up to one coarse voxel off each side — pad it back.
    for e in found.values():
        e['lo'] = e['lo'] - f
        e['hi'] = e['hi'] + f
    print(f"Instances found   : {len(found):,}")
    return found


def read_instance(raw_layer, seg_layer, label: int, lo, hi, max_side: int):
    """Read raw (uint8) + mask for one instance at the finest mag that fits max_side."""
    import webknossos as wk

    longest = int(np.max(hi - lo))
    mag = _pick_mag(seg_layer, int(np.ceil(longest / max_side)))
    box = wk.BoundingBox(tuple(int(v) for v in lo), tuple(int(v) for v in hi - lo))
    box = box.align_with_mag(mag, ceil=True)
    raw = raw_layer.get_mag(mag).read(absolute_bounding_box=box)[0]
    seg = seg_layer.get_mag(mag).read(absolute_bounding_box=box)[0]
    return raw, seg == label, mag.max_dim, np.array(box.topleft.to_tuple())


# ---------------------------------------------------------------------------
# Geometry + projection
# ---------------------------------------------------------------------------

def principal_frame(mask: np.ndarray, max_points: int = 2_000_000):
    """Centroid and principal axes (columns, long → short) of the mask voxels."""
    coords = np.argwhere(mask).astype(np.float64)
    if len(coords) > max_points:
        coords = coords[np.random.default_rng(0).choice(len(coords), max_points, replace=False)]
    centroid = coords.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((coords - centroid).T))
    axes = evecs[:, np.argsort(evals)[::-1]]
    if np.linalg.det(axes) < 0:          # keep it a proper rotation
        axes[:, 2] *= -1
    proj = (coords - centroid) @ axes
    return centroid, axes, proj.min(axis=0), proj.max(axis=0)


def project_instance(raw, mask, zero_level: float, pad: int = 2):
    """
    Resample the masked crop into its principal frame and project it.
    Returns per view (lateral, dorsal): sum projection, max projection, silhouette,
    each shaped (rows=across, cols=long axis), in raw uint8 units above zero_level.
    """
    centroid, axes, lo, hi = principal_frame(mask)
    shape = tuple(int(v) for v in np.ceil(hi - lo) + 1 + 2 * pad)
    # output index o  →  input index  centroid + axes @ (o + lo - pad)
    offset = centroid + axes @ (lo - pad)

    vol = np.clip(raw.astype(np.float32) - zero_level, 0, None)
    vol[~mask] = 0
    vol_r  = ndimage.affine_transform(vol, axes, offset=offset, output_shape=shape, order=1, cval=0.0)
    mask_r = ndimage.affine_transform(mask.astype(np.float32), axes, offset=offset,
                                      output_shape=shape, order=1, cval=0.0) > 0.5
    del vol

    views = {}
    for name, ax in (('lateral', 2), ('dorsal', 1)):
        views[name] = dict(
            sum  = vol_r.sum(axis=ax).T,
            max  = vol_r.max(axis=ax).T,
            sil  = mask_r.any(axis=ax).T,
        )
    extents = hi - lo + 1                 # long, mid, short — in read-mag voxels
    return views, extents, centroid


def render(img: np.ndarray, sil: np.ndarray, vmax: float, scale: float,
           margin: int, invert: bool, darkest: int = 20) -> np.ndarray:
    """Float projection → uint8 image, optional rescale, white (or black) background."""
    if scale < 1.0:
        img = rescale(img, scale, anti_aliasing=True, preserve_range=True)
        sil = rescale(sil.astype(np.float32), scale, anti_aliasing=True, preserve_range=True) > 0.5
    v = np.clip(img / vmax, 0, 1) if vmax > 0 else np.zeros_like(img)
    if invert:
        out = 255 - (255 - darkest) * v
        out[~sil] = 255
    else:
        out = 255 * v
        out[~sil] = 0
    out = np.pad(out, margin, constant_values=255 if invert else 0)
    return out.round().astype(np.uint8)


# ---------------------------------------------------------------------------
# EcoTaxa
# ---------------------------------------------------------------------------

class EcoTaxa:
    """Minimal client for the EcoTaxa REST API (login, upload, import, job polling)."""

    def __init__(self, url: str, token: str = None, user: str = None, password: str = None):
        import httpx
        self.url = url.rstrip('/')
        self.http = httpx.Client(timeout=httpx.Timeout(60.0, write=None, read=600.0))
        if not token:
            r = self.http.post(f"{self.url}/login", json=dict(username=user, password=password))
            r.raise_for_status()
            token = r.json()
        self.http.headers['Authorization'] = f"Bearer {token}"

    def _check(self, r):
        if r.is_error:
            raise RuntimeError(f"EcoTaxa {r.request.method} {r.request.url} → {r.status_code}: {r.text}")
        return r.json()

    def create_project(self, title: str, instrument: str = '?') -> int:
        # access "0" = private; share it from the EcoTaxa UI when ready.
        return self._check(self.http.post(f"{self.url}/projects/create",
                                          json=dict(title=title, instrument=instrument,
                                                    visible=False, access='0')))

    def upload(self, path: Path) -> str:
        with open(path, 'rb') as fh:
            r = self.http.post(f"{self.url}/user_files/",
                               files={'file': (path.name, fh, 'application/zip')},
                               data={'path': path.name})
        return self._check(r)

    def start_import(self, project_id: int, server_path: str) -> int:
        rsp = self._check(self.http.post(f"{self.url}/file_import/{project_id}",
                                         json=dict(source_path=server_path,
                                                   skip_loaded_files=False,
                                                   skip_existing_objects=True)))
        if rsp.get('errors'):
            raise RuntimeError("EcoTaxa rejected the import:\n  " + "\n  ".join(rsp['errors']))
        return rsp['job_id']

    def wait(self, job_id: int, poll: float = 10.0) -> dict:
        while True:
            job = self._check(self.http.get(f"{self.url}/jobs/{job_id}/"))
            state = job['state']
            print(f"  job {job_id}: {state} {job.get('progress_pct') or 0}% {job.get('progress_msg') or ''}")
            if state in ('F', 'E', 'A'):
                return job
            time.sleep(poll)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

TSV_FIELDS = [
    # name,                     type
    ('img_file_name',          't'),
    ('img_rank',               'f'),
    ('object_id',              't'),
    ('object_lat',             'f'),
    ('object_lon',             'f'),
    ('object_date',            't'),
    ('object_wk_label',        'f'),
    ('object_centroid_x',      'f'),
    ('object_centroid_y',      'f'),
    ('object_centroid_z',      'f'),
    ('object_volume_um3',      'f'),
    ('object_length_um',       'f'),
    ('object_width_um',        'f'),
    ('object_thickness_um',    'f'),
    ('object_area_lateral_um2','f'),
    ('object_read_mag',        'f'),
    ('sample_id',              't'),
    ('sample_wk_dataset',      't'),
    ('acq_id',                 't'),
    ('acq_instrument',         't'),
    ('process_id',             't'),
    ('process_projection',     't'),
    ('process_pixel',          'f'),
]


def export(
    sample:       str,
    output_dir:   Path,
    wk_url:       str,
    wk_token:     str,
    dataset:      str   = None,
    raw_layer:    str   = 'raw',
    seg_layer:    str   = 'instances',
    projection:   str   = 'sum',
    norm_min:     float = 0.0,
    norm_max:     float = 1.0,
    pixel_um:     float = None,
    max_side:     int   = 1024,
    scan_mag:     int   = 4,
    min_voxels:   int   = 0,
    labels:       list  = None,
    limit:        int   = None,
    margin:       int   = 10,
    invert:       bool  = True,
    workers:      int   = 8,
    lat:          float = None,
    lon:          float = None,
    date:         str   = None,
) -> Path:
    t0 = time.time()
    dataset = dataset or f"IMATREC_{sample}"
    open_args = (dataset, wk_url, wk_token, raw_layer, seg_layer)
    ds, raw_l, seg_l = open_wk_dataset(*open_args)
    voxel_nm = np.array(ds.voxel_size, dtype=float)
    if not np.allclose(voxel_nm, voxel_nm[0]):
        raise ValueError(f"Anisotropic voxels {tuple(voxel_nm)} are not supported.")
    voxel_um = voxel_nm[0] / 1000.0
    print(f"Dataset '{dataset}'  voxel {voxel_um:.3f} µm  mags {[m.max_dim for m in _available_mags(seg_l)]}")

    # The raw layer was stored as uint8 over [norm_min, norm_max]; this is where
    # attenuation 0 (empty medium) sits on that scale.
    zero_level = float(np.clip(255 * (0 - norm_min) / (norm_max - norm_min), 0, 255))

    # --- Which instances ---
    found = scan_instances(seg_l, scan_mag=scan_mag)
    todo = sorted(l for l, e in found.items() if e['voxels'] >= min_voxels)
    if labels:
        todo = [l for l in todo if l in set(labels)]
    if limit:
        todo = todo[:limit]
    print(f"Exporting {len(todo):,} instances  (projection={projection}, workers={workers})")

    sample_dir = output_dir / sample
    img_dir = sample_dir / 'images'
    tmp_dir = sample_dir / '_proj'
    img_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: read + project every instance; keep float projections on disk ---
    def work(label):
        e = found[label]
        raw_t, seg_t = _thread_layers(*open_args)
        raw, mask, f, origin = read_instance(raw_t, seg_t, label, e['lo'], e['hi'], max_side)
        if mask.sum() < 4:
            return None
        views, extents, centroid = project_instance(raw, mask, zero_level)
        vu = voxel_um * f
        np.savez(tmp_dir / f"{label}.npz",
                 **{f"{v}_{k}": a for v, d in views.items() for k, a in d.items()})
        return dict(
            label     = label,
            mag       = f,
            vu        = vu,
            centroid  = origin + centroid * f,                    # Mag(1) xyz
            volume    = float(mask.sum()) * vu ** 3,
            extents   = extents * vu,
            area      = float(views['lateral']['sil'].sum()) * vu ** 2,
            vmax_sum  = [float(np.percentile(d['sum'][d['sil']], 99.5)) * vu
                         for d in views.values() if d['sil'].any()],
        )

    meta = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for m in tqdm(pool.map(work, todo), total=len(todo), desc='project', unit='obj'):
            if m is not None:
                meta.append(m)

    # --- One intensity window for the whole sample ---
    if projection == 'sum':
        # sum is a line integral (× voxel length) so it is resolution-independent;
        # use the median of per-object 99.5th percentiles so a few very dense
        # objects don't wash out everyone else.
        vmax = float(np.median([v for m in meta for v in m['vmax_sum']])) if meta else 1.0
    else:
        vmax = 255.0 - zero_level        # norm_max on the uint8 scale
    print(f"Intensity window  : 0 – {vmax:.3g}  ({projection})")

    # --- Pass 2: render images + TSV rows ---
    rows = []
    for m in tqdm(meta, desc='render', unit='obj'):
        d = np.load(tmp_dir / f"{m['label']}.npz")
        out_um = max(pixel_um or m['vu'], m['vu'])
        scale = m['vu'] / out_um
        object_id = f"{sample}_{m['label']:05d}"
        for rank, view in enumerate(('lateral', 'dorsal')):
            img = d[f"{view}_{projection}"] * (m['vu'] if projection == 'sum' else 1.0)
            png = render(img, d[f"{view}_sil"], vmax, scale, margin, invert)
            fname = f"{object_id}_{view}.png"
            Image.fromarray(png).save(img_dir / fname)
            L, W, T = m['extents']
            rows.append({
                'img_file_name':           f"images/{fname}",
                'img_rank':                rank,
                'object_id':               object_id,
                'object_lat':              lat,
                'object_lon':              lon,
                'object_date':             date,
                'object_wk_label':         m['label'],
                'object_centroid_x':       round(float(m['centroid'][0]), 1),
                'object_centroid_y':       round(float(m['centroid'][1]), 1),
                'object_centroid_z':       round(float(m['centroid'][2]), 1),
                'object_volume_um3':       round(m['volume'], 1),
                'object_length_um':        round(float(L), 2),
                'object_width_um':         round(float(W), 2),
                'object_thickness_um':     round(float(T), 2),
                'object_area_lateral_um2': round(m['area'], 1),
                'object_read_mag':         m['mag'],
                'sample_id':               sample,
                'sample_wk_dataset':       dataset,
                'acq_id':                  sample,
                'acq_instrument':          'TREC X-ray tomography',
                'process_id':              f"{sample}_{projection}",
                'process_projection':      projection,
                'process_pixel':           round(out_um, 4),
            })
    shutil.rmtree(tmp_dir)

    # Leave out position/date columns that weren't given rather than writing blanks.
    given = dict(object_lat=lat, object_lon=lon, object_date=date)
    fields = [(n, t) for n, t in TSV_FIELDS if given.get(n, 0) is not None]
    tsv = sample_dir / f"ecotaxa_{sample}.tsv"
    with open(tsv, 'w', newline='') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow([n for n, _ in fields])
        w.writerow([f"[{t}]" for _, t in fields])
        for r in rows:
            w.writerow([r[n] for n, _ in fields])

    zip_path = Path(shutil.make_archive(str(output_dir / sample), 'zip', root_dir=sample_dir))
    print(f"\nWrote {len(meta):,} objects / {len(rows):,} images in {time.time() - t0:.1f}s")
    print(f"  TSV : {tsv}")
    print(f"  ZIP : {zip_path}")
    return zip_path


def upload_to_ecotaxa(zip_path: Path, url: str, token: str, user: str, password: str,
                      project_id: int = None, new_project: str = None, instrument: str = '?'):
    eco = EcoTaxa(url, token=token, user=user, password=password)
    if project_id is None:
        project_id = eco.create_project(new_project, instrument=instrument)
        print(f"Created EcoTaxa project {project_id} '{new_project}' (private)")
    print(f"Uploading {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB) ...")
    server_path = eco.upload(zip_path)
    print(f"  on server: {server_path}")
    job_id = eco.start_import(project_id, server_path)
    print(f"Import job {job_id} started into project {project_id}")
    job = eco.wait(job_id)
    if job['state'] == 'F':
        print(f"Import finished: {url.rstrip('/').removesuffix('/api')}/prj/{project_id}")
    elif job['state'] == 'A':
        print(f"Import needs input — answer it in the EcoTaxa UI (Jobs). Question: {job.get('question')}")
    else:
        raise RuntimeError(f"Import job {job_id} failed: {job.get('errors') or job.get('progress_msg')}")
    return project_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Export WebKnossos plankton instances as 2D projections for EcoTaxa.'
    )
    try:
        from config import DEFAULT_CONFIG as _cfg
        _norm_min, _norm_max = _cfg.norm_min, _cfg.norm_max
    except Exception:
        _norm_min, _norm_max = 0.0, 1.0

    g = p.add_argument_group('WebKnossos source')
    g.add_argument('--sample',      required=True,
                   help='Sample name; dataset is IMATREC_<sample> unless --dataset is given.')
    g.add_argument('--dataset',     default=None, help='WebKnossos dataset name (overrides IMATREC_<sample>).')
    g.add_argument('--wk_url',      default='https://webknossos.embl-hamburg.de')
    g.add_argument('--wk_token',    default=os.environ.get('WK_TOKEN'),
                   help='WebKnossos API token (default: WK_TOKEN env var).')
    g.add_argument('--raw_layer',   default='raw')
    g.add_argument('--seg_layer',   default='instances')
    g.add_argument('--norm_min',    type=float, default=_norm_min,
                   help='Window the raw layer was stored with (default: config value).')
    g.add_argument('--norm_max',    type=float, default=_norm_max)

    g = p.add_argument_group('Selection + projection')
    g.add_argument('--output',      required=True, type=Path, help='Output directory.')
    g.add_argument('--projection',  choices=['sum', 'max'], default='sum',
                   help='sum = simulated radiograph (default), max = MIP.')
    g.add_argument('--pixel_um',    type=float, default=None,
                   help='Resample images to this pixel size in µm (e.g. 10.6 to match ZooScan). '
                        'Default: native read resolution.')
    g.add_argument('--max_side',    type=int, default=1024,
                   help='Read each instance at the finest mag where its bbox fits this many voxels.')
    g.add_argument('--scan_mag',    type=int, default=4, help='Mag for the bounding-box scan.')
    g.add_argument('--min_voxels',  type=int, default=0, help='Skip instances smaller than this (Mag 1 voxels).')
    g.add_argument('--labels',      type=int, nargs='+', default=None, help='Only these instance labels.')
    g.add_argument('--limit',       type=int, default=None, help='At most this many instances (for testing).')
    g.add_argument('--margin',      type=int, default=10, help='Background border in pixels.')
    g.add_argument('--no_invert',   action='store_true',
                   help='Bright object on black (UVP-like) instead of dark on white (ZooScan-like).')
    g.add_argument('--workers',     type=int, default=8, help='Parallel instance downloads.')
    g.add_argument('--lat',         type=float, default=None, help='Sampling latitude (decimal degrees).')
    g.add_argument('--lon',         type=float, default=None, help='Sampling longitude (decimal degrees).')
    g.add_argument('--date',        default=None, help='Sampling date, YYYYMMDD.')

    g = p.add_argument_group('EcoTaxa upload')
    g.add_argument('--upload',      action='store_true', help='Upload the zip and import it into EcoTaxa.')
    g.add_argument('--ecotaxa_url', default='https://ecotaxa.obs-vlfr.fr/api')
    g.add_argument('--project_id',  type=int, default=None, help='Existing EcoTaxa project to import into.')
    g.add_argument('--new_project', default=None, help='Create a new private project with this title.')
    g.add_argument('--instrument',  default='?', help='Instrument for a new project (default: ?).')
    args = p.parse_args()

    if not args.wk_token:
        p.error("Need --wk_token or the WK_TOKEN environment variable.")
    eco_token = os.environ.get('ECOTAXA_TOKEN')
    eco_user  = os.environ.get('ECOTAXA_USER')
    eco_pass  = os.environ.get('ECOTAXA_PASSWORD')
    if args.upload:
        if (args.project_id is None) == (args.new_project is None):
            p.error("--upload needs exactly one of --project_id or --new_project.")
        if not eco_token and not (eco_user and eco_pass):
            p.error("--upload needs ECOTAXA_TOKEN, or ECOTAXA_USER + ECOTAXA_PASSWORD, in the environment.")

    zip_path = export(
        sample     = args.sample,
        output_dir = args.output,
        wk_url     = args.wk_url,
        wk_token   = args.wk_token,
        dataset    = args.dataset,
        raw_layer  = args.raw_layer,
        seg_layer  = args.seg_layer,
        projection = args.projection,
        norm_min   = args.norm_min,
        norm_max   = args.norm_max,
        pixel_um   = args.pixel_um,
        max_side   = args.max_side,
        scan_mag   = args.scan_mag,
        min_voxels = args.min_voxels,
        labels     = args.labels,
        limit      = args.limit,
        margin     = args.margin,
        invert     = not args.no_invert,
        workers    = args.workers,
        lat        = args.lat,
        lon        = args.lon,
        date       = args.date,
    )

    if args.upload:
        upload_to_ecotaxa(zip_path, args.ecotaxa_url, eco_token, eco_user, eco_pass,
                          project_id=args.project_id, new_project=args.new_project,
                          instrument=args.instrument)


if __name__ == '__main__':
    main()
