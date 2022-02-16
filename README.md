# alibi-detect-clean
    The clean version of alibi-test.
    Only contains two-sample KS test and ChiSquare test to detect distribution drift of two samples.
    This version suppurts Tensorflow and Pytorch which allows the model to run on GPUs.

### How to use:
    from drift_detection import drift_detection
    ref_logist = my_model(X_ref)
    drift_detector = drift_detection(ref_logist, 0.01, method="KSDrift")
    or
    drift_detector = drift_detection(ref_logist, 0.01, method="ChiSquareDrift")
    
    you can update any parameters and data by
    drift_detector.update_ref(new_x_ref)
    drift_detector.update_threshold(new_threshold)
    drift_detector.update_return_p_val(new_return_p_val)
    drift_detector.update_return_distance(new_return_distance)
    
    get the result by 
    x_logist = my_model(x)
    preds = drift_detector.get_result(x_logist)
