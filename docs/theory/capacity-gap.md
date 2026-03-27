# Capacity Gap (Han et al.)

*Status: Established (external). Han et al. 2024, arXiv:2412.06590.*

Han et al. prove that softmax attention is injective (different queries produce different distributions) while linear attention is not (distinct queries can collapse to identical outputs). This capacity gap is the structural basis for using softmax-linear divergence as a diagnostic.

SAD does not claim that divergence directly measures truth --- it measures how much the model relies on its full nonlinear attention capacity versus operating in a regime where the weaker linear mechanism suffices.

<!-- Phase 2: Formal statement of the injectivity result, implications for SAD design, why this makes cosine divergence informative -->
