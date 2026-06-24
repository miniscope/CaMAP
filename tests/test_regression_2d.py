"""Full-pipeline regression test for the 2D arena analysis.

Runs the complete ArenaDataset pipeline on a small data subset and
compares every output against a saved reference bundle.

To regenerate the reference bundle with low ``n_shuffles`` in the config::

    camap analysis -c config.yaml -d data.yaml -o tests/assets/regression_2d/reference -y --subset-units 10 --subset-frames 10000
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from camap.dataset.arena import ArenaDataset
from camap.dataset.base import BaseCaMAPDataset


REGRESSION_DIR = Path(__file__).parent / "assets" / "regression_2d"


def _renamed_columns_data_paths(src_dir: Path, dst_dir: Path) -> Path:
    """Build a data_paths.yaml whose timestamp CSVs use non-default columns.

    Copies the neural and behavior timestamp CSVs into *dst_dir* with their
    canonical headers (``timestamp_first``, ``frame_index``, ``unix_time``)
    renamed, then writes a data config that points at those copies and sets
    the ``timestamp_*_col`` fields. All other inputs (zarr, position CSV) are
    referenced in place via absolute paths. Returns the new data_paths.yaml.
    """
    neural_ts = pd.read_csv(src_dir / "neural_timestamp.csv").rename(
        columns={"timestamp_first": "ts_first"}
    )
    neural_ts.to_csv(dst_dir / "neural_timestamp.csv", index=False)
    beh_ts = pd.read_csv(src_dir / "behavior_timestamp.csv").rename(
        columns={"frame_index": "f_idx", "unix_time": "t_unix"}
    )
    beh_ts.to_csv(dst_dir / "behavior_timestamp.csv", index=False)

    cfg = yaml.safe_load((src_dir / "data_paths.yaml").read_text())
    cfg["neural"]["path"] = str(src_dir / "neural_data")
    cfg["neural"]["timestamp"] = str(dst_dir / "neural_timestamp.csv")
    cfg["neural"]["timestamp_col"] = "ts_first"
    cfg["behavior"]["position"] = str(src_dir / "behavior_position.csv")
    cfg["behavior"]["timestamp"] = str(dst_dir / "behavior_timestamp.csv")
    cfg["behavior"]["timestamp_frame_col"] = "f_idx"
    cfg["behavior"]["timestamp_time_col"] = "t_unix"

    out = dst_dir / "data_paths.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


@pytest.fixture(scope="module", params=["canonical", "renamed_columns"])
def pipeline_result(request: pytest.FixtureRequest, tmp_path_factory) -> ArenaDataset:
    """Run the full pipeline once per data-config variant.

    ``canonical`` uses the default timestamp column names; ``renamed_columns``
    points the loader at CSVs with non-default headers via the
    ``timestamp_*_col`` config fields, exercising both neural and behavior
    timestamp customization end to end. Both must produce identical results.
    """
    if request.param == "canonical":
        data_paths = REGRESSION_DIR / "data_paths.yaml"
    else:
        tmp = tmp_path_factory.mktemp("renamed_columns")
        data_paths = _renamed_columns_data_paths(REGRESSION_DIR, tmp)

    ds = BaseCaMAPDataset.from_yaml(
        REGRESSION_DIR / "analysis_config.yaml",
        data_paths,
    )
    ds.load()
    ds.preprocess_behavior()
    ds.deconvolve()
    ds.match_events()
    ds.compute_occupancy()
    ds.analyze_units()
    return ds


@pytest.fixture(scope="module")
def reference() -> ArenaDataset:
    """Load the reference bundle."""
    return BaseCaMAPDataset.load_bundle(REGRESSION_DIR / "reference.camap")



@pytest.mark.timeout(120)
def test_summary_counts(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Pipeline summary counts must match the reference."""
    assert pipeline_result.summary() == reference.summary()



def test_good_unit_ids(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Deconvolved unit IDs must match."""
    assert pipeline_result.good_unit_ids == reference.good_unit_ids


def test_event_index_shape(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Non-trivial event count must match.

    OASIS deconvolution can produce near-zero ghost events (s ~ 1e-17)
    whose presence varies across platforms.  Filter these out before
    comparing counts.
    """
    threshold = 1e-10
    got = (pipeline_result.event_index["s"] > threshold).sum()
    ref = (reference.event_index["s"] > threshold).sum()
    assert got == ref



def test_event_place_shape(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Non-trivial matched event count must match (see event_index note)."""
    threshold = 1e-10
    got = (pipeline_result.event_place["s"] > threshold).sum()
    ref = (reference.event_place["s"] > threshold).sum()
    assert got == ref



def test_occupancy_map(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Occupancy map must match reference."""
    np.testing.assert_allclose(
        pipeline_result.occupancy_time,
        reference.occupancy_time,
        rtol=1e-5,
    )


def test_valid_mask(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Valid mask must match reference."""
    np.testing.assert_array_equal(
        pipeline_result.valid_mask,
        reference.valid_mask,
    )



def test_unit_result_ids(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Analyzed unit IDs must match."""
    assert sorted(pipeline_result.unit_results.keys()) == sorted(
        reference.unit_results.keys()
    )


def test_unit_scalars(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Per-unit scalar metrics (SI, p_val, stability) must match."""
    for uid in reference.unit_results:
        ref = reference.unit_results[uid]
        got = pipeline_result.unit_results[uid]

        assert got.si == pytest.approx(ref.si, rel=1e-5), f"unit {uid} SI"
        assert got.p_val == pytest.approx(ref.p_val, rel=1e-5), f"unit {uid} p_val"
        assert len(got.stability_splits) == len(ref.stability_splits), (
            f"unit {uid} stability_splits length"
        )
        for i, (g, r) in enumerate(zip(got.stability_splits, ref.stability_splits)):
            assert g.n_split_blocks == r.n_split_blocks, f"unit {uid} split {i} n_split_blocks"
            assert g.corr == pytest.approx(r.corr, nan_ok=True, rel=1e-5), (
                f"unit {uid} split {i} corr"
            )
            assert g.fisher_z == pytest.approx(r.fisher_z, nan_ok=True, rel=1e-5), (
                f"unit {uid} split {i} fisher_z"
            )
            assert g.p_val == pytest.approx(r.p_val, nan_ok=True, rel=1e-5), (
                f"unit {uid} split {i} p_val"
            )


def test_rate_maps(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Per-unit rate maps must match."""
    for uid in reference.unit_results:
        ref_map = reference.unit_results[uid].rate_map_smoothed
        got_map = pipeline_result.unit_results[uid].rate_map_smoothed
        assert got_map.shape == ref_map.shape, f"unit {uid} rate_map shape"
        np.testing.assert_allclose(
            got_map,
            ref_map,
            rtol=1e-5,
            atol=1e-10,
            equal_nan=True,
            err_msg=f"unit {uid} rate_map_smoothed",
        )



def test_save_load_bundle_roundtrip(
    pipeline_result: ArenaDataset,
) -> None:
    """save_bundle → load_bundle must round-trip without error and preserve results."""
    with tempfile.TemporaryDirectory() as tmp:
        bundle_path = pipeline_result.save_bundle(
            Path(tmp) / "test", save_figures=False
        )
        reloaded = BaseCaMAPDataset.load_bundle(bundle_path)

    assert isinstance(reloaded, ArenaDataset)
    assert reloaded.summary() == pipeline_result.summary()
    assert sorted(reloaded.unit_results.keys()) == sorted(
        pipeline_result.unit_results.keys()
    )
    for uid in pipeline_result.unit_results:
        np.testing.assert_allclose(
            reloaded.unit_results[uid].rate_map_smoothed,
            pipeline_result.unit_results[uid].rate_map_smoothed,
            rtol=1e-5,
            equal_nan=True,
            err_msg=f"unit {uid} rate_map_smoothed round-trip",
        )
