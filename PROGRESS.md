# Progress Meeting

## Time: June 2

Yingte: I tried to reproduce the MiniWorld experiments.
We did a grid search: on the following setup:
- Environment: Same geometry, with clean / action_noise / noise-tv
- Algorithms: LPM, RND, ICM, MSE, None.
- ten seeds for each run.
5 * 3 * 10 = 150 runs in total.

Discoveries:
- The TV-action share of LPM is the lowest, way below the uniform baseline. Its coverage is larger than other curiosity based exploration methods.
- But suprisingly, the None baseline (no intrinsic reward) seems to be performing better.
- Curiosity driven methods have very large variance, compared to the baseline without intrinsic reward.
- The MiniWorld evaluation metric is coverage, but it explains that LPM avoids the problem of being distracted by noisy-TV, but it is not necessarily the best exploration method. In fact, a uniform distribution policy on actions will beat all methods. This leads to the question: is intrinsic reward really a useful trick? In what scenario?
- Intrinsic Reward: 

## Time: May 30

Yingte: I discovered a interesting question: In the Atari game, LPM is performing better in the clean Ms PacMan setting. Theoretically speaking, LPM should not have the advantage in the clean setting. The reason may be that the random movement of Ghost can be considered as kind of randomness or noise. But this is worth exploring.

Other kinds of measures: Understand the exploration evolvement in 3D maze.

## Time: May 22

Our current plan for the project:

1. First-stage: demonstrate exploration ability difference of different learning methods.
2. Try out LPM on an environment with real environment with extrinsic reward, and compare with other solutions.

We decide to split the work in this way:

Yingte:
1. Look into the paper deeper and understand the formulas.
2. Find the suitable environment for further investigation.

Youssef:
1. Write a summary.
2. Implement LPM on the miniworld environment, and observe the exploration behavior.

Next meeting: