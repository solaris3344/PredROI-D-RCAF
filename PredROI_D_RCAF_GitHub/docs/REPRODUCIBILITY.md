# Reproducibility notes

The paper uses a fixed patient-disjoint five-fold Cheng evaluation with outer seed 5813. Localizer cross-fitting is patient-disjoint. The predicted-box rule uses a primary threshold of 0.50, one fallback threshold of 0.30, and a square expansion factor of 1.10. The main classifier uses Swin-Base at 224 x 224, with a 15-epoch classifier-selection ceiling and the encoder frozen for the first three epochs in the PredROI/fusion experiments.

For stochastic sensitivity, patient membership is held fixed while only the run seed changes. The reported additional run seeds are 7253, 9341, and 12011.

External BDNeuro-MRI results are image-level because patient identifiers are unavailable. The external PredROI result in the manuscript is a protocol reconstruction rather than a bitwise replay of the original locked PredROI model because the original final fold classifier weights were not retained.

The `reviewer_closure/` folder contains post-lock audit/closure analyses. These should be described as secondary or exploratory when appropriate, not as prespecified confirmatory tests.
