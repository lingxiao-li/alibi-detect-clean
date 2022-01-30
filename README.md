# alibi-detect-clean
The clean version of alibi-test.
Only contains two-sample KS test to detect distribution drift of two samples.

### How to use:
    
    from alibi_detect.cd.model_uncertainty import ClassifierUncertaintyDrift
    
  	cd = ClassifierUncertaintyDrift(
    X_ref#X_c[0][0:500], model=my_model, backend='pytorch', p_val=0.05, preds_type='logits'
    )
