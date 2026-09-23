"""Independent mathematical and computational checks of the production contract."""
from itertools import combinations
import numpy as np
import pytest
from scipy.stats import wasserstein_distance, false_discovery_control

import wade
from wade._wasserstein import HAVE_RUST_KERNEL, TIE_RTOL, W1State, infer
from wade.pvalues import sequential_p

BACKENDS = ["numpy"] + (["rust"] if HAVE_RUST_KERNEL else [])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("sizes", [(2, 2), (3, 17), (50, 500)])
def test_exact_distance_and_stopping_against_scipy(backend, sizes):
    n1, n0 = sizes
    r = np.random.default_rng(71)
    x = np.r_[np.zeros((1,n1+n0)), np.log2(r.integers(0, 12, (7,n1+n0))+1)]
    cond = np.r_[np.ones(n1,int),np.zeros(n0,int)]
    labels = wade.draw_perms(cond, 79, seed=88)
    state = W1State(x,cond,stop=7)
    state.update(labels[:11],backend=backend)
    state.update(labels[11:],backend=backend)
    for j,row in enumerate(x):
        observed = wasserstein_distance(row[:n1],row[n1:])
        null = np.array([wasserstein_distance(row[p==1],row[p==0]) for p in labels])
        hits = null >= observed*(1-TIE_RTOL)
        crossing = np.flatnonzero(hits)
        used = crossing[6]+1 if len(crossing)>=7 else len(labels)
        assert state.observed[j] == pytest.approx(observed,abs=1e-13)
        assert state.tried[j] == used
        assert state.exceed[j] == hits[:used].sum()
    assert state.tried[0] == 7
    assert sequential_p(state.exceed,state.tried,stop=7,maximum=79)[0] == 1


@pytest.mark.validation
def test_sequential_rule_is_superuniform_at_every_attainable_pvalue():
    for cap in (3, 8, 31, 499):
        for stop in (1, 2, min(20,cap)):
            t=np.arange(stop,cap+1)
            k=np.arange(stop)
            mass=np.r_[stop/(t*(t+1)),np.full(stop,1/(cap+1))]
            p=sequential_p(np.r_[np.full(len(t),stop),k],np.r_[t,np.full(stop,cap)],stop=stop,maximum=cap)
            assert mass.sum()==pytest.approx(1)
            for alpha in np.unique(p):
                assert mass[p<=alpha].sum() <= alpha+2e-14


@pytest.mark.parametrize("hits,draws", [([1],[3]),([21],[39]),([-1],[39]),([20],[10]),([.5],[39]),([np.nan],[39])])
def test_incomplete_records_cannot_be_reported(hits,draws):
    with pytest.raises(ValueError):
        sequential_p(hits,draws,stop=20,maximum=39)


@pytest.mark.parametrize("backend", BACKENDS)
def test_exhaustive_small_orbit_is_symmetric_and_calibrated(backend):
    x=np.array([[0,0,0,2,4,4],[0,0,0,0,0,100],[3,3,3,3,3,3]],float)
    orbit=[]
    for idx in combinations(range(6),3):
        p=np.zeros(6,np.uint8); p[list(idx)]=1; orbit.append(p)
    orbit=np.array(orbit)
    stats=np.array([W1State(x,p).observed for p in orbit]).T
    for row in stats:
        p=(row[None,:]>=row[:,None]*(1-TIE_RTOL)).mean(axis=1)
        for alpha in np.unique(p):
            assert np.mean(p<=alpha)<=alpha+1e-14
    np.testing.assert_allclose(stats[:,0],W1State(x,1-orbit[0]).observed)


@pytest.mark.parametrize("backend", BACKENDS)
def test_chunk_batch_strata_and_filter_invariance(backend):
    rng=np.random.default_rng(85)
    x=rng.poisson(30,(30,20)); x[0]=0; x[1,:10]*=8
    cond=np.tile([1,0],10); strata=np.repeat(np.arange(4),5)
    kwargs=dict(max_permutations=97,seed=17,strata=strata,backend=backend)
    a=wade.wade(x,cond,gene_chunk=30,**kwargs)
    b=wade.wade(x,cond,gene_chunk=3,**kwargs)
    for key in ("statistic","pvalue","padj","n_permutations","n_exceedances"):
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    mask=np.arange(30)%2==0
    c=wade.wade(x,cond,test_genes=mask,**kwargs)
    np.testing.assert_array_equal(a.normalization.size_factors,c.normalization.size_factors)
    np.testing.assert_array_equal(a.pvalue[mask],c.pvalue[mask])
    np.testing.assert_allclose(c.padj[c.tested],false_discovery_control(c.pvalue[c.tested]))
    for batch in (1,13,97):
        out=infer(a.expression[1:3],cond,maximum=97,seed=17,strata=strata,backend=backend,batch=batch)
        np.testing.assert_array_equal(out[1],a.pvalue[1:3])
        np.testing.assert_array_equal(out[3],a.n_permutations[1:3])


@pytest.mark.kernel
@pytest.mark.skipif(not HAVE_RUST_KERNEL,reason="native kernel not built")
def test_native_boundary_rejects_invalid_indices():
    from wade import _kernel
    state=W1State(np.ones((2,6)),[1,1,1,0,0,0])
    state.order[0,0]=100
    with pytest.raises(ValueError):
        _kernel.update_w1(state.order,state.gaps,wade.draw_perms([1,1,1,0,0,0],2),state.thresholds,
                          state.exceed,state.tried,3,20)


def test_qvalues_have_one_family_and_excluded_genes_have_no_test():
    x=np.random.default_rng(53).poisson(30,(15,12)); x[0]=0
    result=wade.wade(x,[1]*6+[0]*6,max_permutations=39,test_genes=["gene0","gene2","gene3"])
    assert list(result.status[:4])==["all_zero","filtered","tested","tested"]
    assert np.isnan(result.pvalue[~result.tested]).all()
    assert (result.n_permutations[~result.tested]==0).all()
    np.testing.assert_allclose(result.padj[result.tested],false_discovery_control(result.pvalue[result.tested]))
    assert not hasattr(result,"p_subset")
    assert list(result.columns())==list(wade.RESULT_COLUMNS)


def test_opposing_tails_do_not_cancel_and_description_has_no_significance():
    from wade._descriptions import describe
    x=np.array([[0,0,4,4,2,2,2,2]],float)
    cond=np.array([1]*4+[0]*4)
    assert W1State(x,cond).observed[0]==2
    bulk,upper,lower=describe(x,cond)
    np.testing.assert_allclose([bulk[0],upper[0],lower[0]],[0,2,-2])


def test_quantile_detail_integrates_to_exact_w1_with_unequal_sizes():
    from wade.diagnostics import gene_detail
    x=np.array([0,0,1,10,2,3,4],float); cond=np.array([1,1,1,1,0,0,0])
    detail=gene_detail(x,cond)
    assert detail.cumulative_area[-1]==pytest.approx(wasserstein_distance(x[:4],x[4:]))
    assert np.sum(detail.effect*np.diff(detail.probability))==pytest.approx(x[:4].mean()-x[4:].mean())


@pytest.mark.parametrize("kwargs",[{"max_permutations":0},{"max_permutations":2.5},{"gene_chunk":0},{"backend":"wrong"}])
def test_invalid_settings_are_errors(kwargs):
    with pytest.raises(ValueError):
        wade.wade(np.ones((3,4)),[1,1,0,0],**kwargs)


def test_no_exchange_or_single_sample_design_is_rejected():
    x=np.ones((3,4))
    with pytest.raises(ValueError,match="exchanges"):
        wade.wade(x,[1,1,0,0],strata=[1,1,0,0])
    with pytest.raises(ValueError,match="two independent"):
        wade.wade(x,[1,0,0,0])


def test_restricted_shuffle_preserves_each_group_and_missing_strata_rejected():
    from wade.permutation import validate_perms
    cond=np.array([1,1,0,1,0,0]); strata=np.repeat(["a","b"],3)
    labels=wade.draw_perms(cond,101,seed=42,strata=strata)
    validate_perms(labels,cond,101,strata=strata)
    assert (labels[:,:3].sum(axis=1)==2).all()
    assert (labels[:,3:].sum(axis=1)==1).all()
    with pytest.raises(ValueError,match="missing"):
        wade.draw_perms(cond,3,strata=["a"]*5+[None])
