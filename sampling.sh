#["rtts", "urhi", 'etc', 'custom']
# Add --config.eval.compute_iqa True if you also want BRISQUE / NIMA / MUSIQ / PAQ2PIQ.
python ./dehaze_sampling.py \
    --config ./configs/rectified_flow/cifar10_rf_gaussian.py  \
    --config.data.dataset "custom" \
    --config.data.test_data_root "datasets/custom" \
    --config.flow.refine_t True \
    --config.sampling.ckpt 'hazeflow.pth' \
    --config.work_dir 'samples/' \
    --config.expr 1step \
    --config.sampling.sample_N 1 \
