import json
import numpy as np
import pytest
import wade


@pytest.fixture
def result():
    x=np.random.default_rng(22).poisson(20,(15,12)); x[0]=0; x[2,:3]*=30; x[3,6:]=0
    return wade.wade(x,[1]*6+[0]*6,max_permutations=41)


@pytest.mark.parametrize("extension",["tsv","csv","parquet","arrow","feather"])
def test_export_manifest_and_table_roundtrip(result,tmp_path,extension):
    pl=pytest.importorskip("polars")
    path=wade.write_results(result,tmp_path/f"report.{extension}")
    if extension in ("tsv","csv"):
        frame=pl.read_csv(path,separator="\t" if extension=="tsv" else ",")
    elif extension=="parquet":
        frame=pl.read_parquet(path)
    else:
        frame=pl.read_ipc(path)
    for column in ("case_mean","area","area_fraction","log2fc","area_padj","subset_log2fc","subset_padj","subset_rarity"):
        np.testing.assert_allclose(frame[column].to_numpy(),getattr(result,column),equal_nan=True)
    assert frame.columns==list(wade.RESULT_COLUMNS)
    record=json.loads((tmp_path/'report.manifest.json').read_text())
    assert record["design"]["n_tested"]==14
    assert record["resampling"]["area_permutations"]==result.area_permutations.tolist()
    assert record["run"]["depth_equalized"]==result.params["depth_equalized"]
    assert record["results"]["hypotheses"]==14 and record["run"]["direction"]=="both"
    assert set(record["run"]["size_factor_median"])=={"case","control"}
    assert record["run"]["max_permutations"]==41
    assert record["normalization"]["used_reference_genes"]==list(result.normalization.used_reference_genes)


@pytest.mark.parametrize("backend",["matplotlib","plotly"])
@pytest.mark.parametrize("theme",["light","dark","high-contrast"])
def test_plots_render_from_one_schema(result,tmp_path,backend,theme):
    pytest.importorskip(backend)
    for name,args,kwargs in [("plot_gene",("gene2",),{}),("plot_gene",("gene0",),{}),
                             ("plot_volcano",(),{"label":2}),("plot_patterns",(),{"label":["gene2"]})]:
        fig=getattr(result,name)(*args,backend=backend,theme=theme,**kwargs)
        if backend=="matplotlib":
            import matplotlib.pyplot as plt
            fig.savefig(tmp_path/f"{name}.png",dpi=90)
            plt.close(fig)
        else:
            fig.to_json()


def test_write_results_takes_integer_gene_ids_and_numpy_labels(tmp_path):
    pl = pytest.importorskip("polars")
    x = np.random.default_rng(3).poisson(20, (6, 8))
    sheet = {"id": [f"s{j}" for j in range(8)], "group": np.r_[np.ones(4, np.int64), np.zeros(4, np.int64)]}
    cond = wade.condition(sheet, key="id", column="group", case=np.int64(1), control=np.int64(0))
    r = wade.wade(x, cond, gene_names=np.arange(100, 106), sample_names=sheet["id"], fdr=np.float32(0.1),
                  max_permutations=99)
    for extension in ("tsv", "parquet"):
        path = wade.write_results(r, tmp_path / f"r.{extension}")
        frame = pl.read_csv(path, separator="\t") if extension == "tsv" else pl.read_parquet(path)
        assert frame["gene"].to_list() == list(range(100, 106))
    record = json.loads((tmp_path / "r.manifest.json").read_text())
    assert record["run"]["case_label"] == 1 and record["run"]["fdr"] == pytest.approx(0.1)


def test_rarity_views_check_their_arguments(result):
    with pytest.raises(ValueError, match="n must"):
        result.rare_subsets(n=0)
    with pytest.raises(ValueError, match="drivers"):
        result.sample_load(drivers=-1)


def test_manual_counts_container_is_validated():
    invalid=wade.Counts(np.array([[1,np.nan,2,2]]),np.array(["g"]),np.array(list("abcd")))
    with pytest.raises(ValueError,match="non-finite"):
        wade.wade(invalid,[1,1,0,0],max_permutations=20)
