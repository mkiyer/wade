import json
import numpy as np
import pytest
import wade


def test_reference_ids_rle_formula_and_json_roundtrip():
    x=np.array([[10,20,30,40],[5,0,10,20],[100,120,90,110]],float)
    norm=wade.fit_normalization(x,reference_genes=["gene0","gene1"])
    assert norm.reference_genes==("gene0","gene1")
    assert norm.used_reference_genes==("gene0",)
    expected=x[0]/np.exp(np.log(x[0]).mean())
    np.testing.assert_allclose(norm.size_factors,expected,rtol=1e-14)
    restored=wade.Normalization.from_dict(json.loads(json.dumps(norm.to_dict())))
    np.testing.assert_array_equal(restored.size_factors,norm.size_factors)
    assert not norm.size_factors.flags.writeable
    np.testing.assert_array_equal(norm.transform(x),x/norm.size_factors)


def test_filtering_does_not_refit_and_reference_test_overlap_is_explicit():
    x=np.random.default_rng(44).poisson(10,(20,12)); x[2,:6]*=10
    norm=wade.fit_normalization(x,reference_genes=["gene0","gene1"])
    mask=wade.filter_counts(x,min_count=3,min_samples=2,min_total=20)
    mask[0]=False
    res=wade.wade(x,[1]*6+[0]*6,normalization=norm,test_genes=mask,max_permutations=31)
    assert res.normalization is norm
    assert wade.manifest(res)["design"]["tested_reference_genes"]==["gene1"]
    np.testing.assert_array_equal(mask,wade.filter_counts(x[:,::-1],min_count=3,min_samples=2,min_total=20)&(np.arange(20)!=0))


def test_supplied_factors_are_centered_once_and_align_by_name():
    x=wade.as_counts(np.arange(1,25).reshape(4,6),sample_names=list("abcdef"))
    norm=wade.Normalization.from_size_factors(x,dict(zip("abcdef",[1,2,3,4,5,6])),source="validated external fit")
    selected=norm.select_samples(["f","b","c","d"])
    np.testing.assert_array_equal(selected.size_factors,norm.size_factors[[5,1,2,3]])
    res=wade.wade_contrast(x,["f","b"],["c","d"],normalization=norm,max_permutations=21)
    np.testing.assert_array_equal(res.normalization.size_factors,selected.size_factors)
    counts=x.values[:,[5,1,2,3]]
    assert np.all((counts/selected.size_factors<=res.expression)&(res.expression<(counts+1)/selected.size_factors))
    assert res.normalization.fit_sample_names==tuple("abcdef")


def test_per_sample_cpm_tpm_and_sparse_zeros():
    x=np.array([[10,0],[0,30],[30,10]],float); lengths=np.array([1,2,3])
    for func,args in [(wade.cpm,()),(wade.tpm,(lengths,))]:
        full=func(x,*args)
        assert (full[x==0]==0).all()
        np.testing.assert_allclose(full.sum(axis=0),1e6)
        for j in range(2):
            np.testing.assert_array_equal(full[:,j],func(x[:,j:j+1],*args)[:,0])


def test_no_silent_fallback_and_missing_control_ids_fail():
    with pytest.raises(ValueError,match="no usable"):
        wade.fit_normalization([[1,0],[0,1]])
    with pytest.raises(ValueError,match="unknown"):
        wade.fit_normalization(np.ones((3,4)),reference_genes=["missing"])
    with pytest.raises(ValueError):
        wade.fit_normalization(np.ones((3,4)),method="rle-log1p")
    with pytest.raises(ValueError):
        wade.fit_normalization([[1,0],[1,0]])
