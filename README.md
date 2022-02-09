# alibi-detect-clean
    The clean version of alibi-test.
    Only contains two-sample KS test and ChiSquare test to detect distribution drift of two samples.
    This version suppurts Tensorflow and Pytorch which allows the model to run on GPUs.

### How to use:
    First use model to predict the label
    ref_logist = my_model(X_ref)
    x_logist = my_model(x_h0)
    Then call the function drift_detection, and select "KSDrift" or "ChiSquareDrift" in the method
    preds = drift_detection(ref_logist, x_logist, 0.01, method="KSDrift")
