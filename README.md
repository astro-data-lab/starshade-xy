# Offset estimation for Starshade Formation Flying

Starshade + Roman Space Telescope coordinated flight needs to be better than 1m precision in lateral diredction, with the only way of assessing good co-location being the image of the obscured pupil as observed from Roman.

The current estimation procedure by JPL uses a large precomputed template library to match the offsets.
This is memory-intensive and probably requires tree-based search for the nearest template to the given observation.
It is also dependent on the star observed (because of the chromaticity of the diffraction pattern).

A simple CNN should do pretty well there: `Image -> CNN -> (dx, dy, t)`, where `t` is the spectral type of the stars

The patterns is pseudo-linear in the offsets in the inner region, but gets more complication further away.
The transition region is currently not well modeled at all.
The pattern is very smooth (complex shape seen from a large distance with an almost perfectly collimated beam) and can be simulated well.
The chromatic effects are limited because of the narrow wavelength range use used for control. 
Thus, `Image -> CNN -> (dx,dy)`should work really well. 
One caveat is SNR of the star might differ (in the narrow band), so training should include SNR variations.

Also, because the actual image in flight (or on the testbed) might be subtly different from the simulations, we should predict a quality flag or uncertainty estimate of `dx,dy`, either directly from as CNN output, or in the form of a Bayesian CNN.

## Project outline

1. Train and validate on simulated images (code from Anthony)
2. Test on testbed images (includes air disturbances), will show how robust estimation is
3. If 2. is successful, run (dx,dy) in closed-loop control system on testbed
