# alibi-detect-clean
    The clean version of alibi-test.
    Only contains two-sample KS test to detect distribution drift of two samples.
    This version suppurts Tensorflow and Pytorch which allows the model to run on GPUs.

### How to use:
    
    from alibi_detect_clean.cd.model_uncertainty import ClassifierUncertaintyDrift
    
  	cd = ClassifierUncertaintyDrift(
    X_ref#X_c[0][0:500], model=my_model, backend='pytorch', p_val=0.05, preds_type='logits'
    )
