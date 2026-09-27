import json
import numpy as np
import pytest
import wade


@pytest.fixture
def result():
    x=np.random.default_rng(22).poisson(20,(15,12)); x[0]=0; x[2,:3]*=30
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
    for column in ("padj_up","padj_down","area_up"):
        np.testing.assert_allclose(frame[column].to_numpy(),getattr(result,column),equal_nan=True)
    assert frame.columns==list(wade.RESULT_COLUMNS)
    record=json.loads((tmp_path/'report.manifest.json').read_text())
    assert record["design"]["n_tested"]==14
    assert record["resampling"]["n_permutations"]==result.n_permutations.tolist()
    assert record["results"]["hypotheses"]==28 and record["run"]["direction"]=="both"
    assert set(record["run"]["size_factor_median"])=={"case","control"}
    assert record["run"]["max_permutations"]==41
    assert record["normalization"]["used_reference_genes"]==list(result.normalization.used_reference_genes)


@pytest.mark.parametrize("backend",["matplotlib","plotly"])
@pytest.mark.parametrize("theme",["light","dark","high-contrast"])
def test_plots_render_from_one_schema(result,tmp_path,backend,theme):
    pytest.importorskip(backend)
    for name,args,kwargs in [("plot_gene",("gene2",),{}),("plot_volcano",(),{}),
                             ("plot_volcano",(),{"direction":"down"}),("plot_patterns",(),{"direction":"down"})]:
        fig=getattr(result,name)(*args,backend=backend,theme=theme,**kwargs)
        if backend=="matplotlib":
            import matplotlib.pyplot as plt
            fig.savefig(tmp_path/f"{name}.png",dpi=90)
            plt.close(fig)
        else:
            fig.to_json()


def test_a_one_direction_result_refuses_to_plot_the_other(result):
    up=wade.wade(np.random.default_rng(22).poisson(20,(6,12)),[1]*6+[0]*6,direction="up",max_permutations=19)
    with pytest.raises(ValueError,match="not tested"):
        up.plot_volcano(direction="down")


def test_manual_counts_container_is_validated():
    invalid=wade.Counts(np.array([[1,np.nan,2,2]]),np.array(["g"]),np.array(list("abcd")))
    with pytest.raises(ValueError,match="non-finite"):
        wade.wade(invalid,[1,1,0,0],max_permutations=20)
