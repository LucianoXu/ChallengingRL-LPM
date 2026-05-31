# Methods status (Ms Pac-Man exploration comparison)

All 6 methods reach the training loop and log sane episode scores (smoke = 25k steps, MsPacmanNoFrameskip-v4):

| label   | --algo          | extra        | runs? | note |
|---------|-----------------|--------------|-------|------|
| lpm     | ppo-improvement | --beta 1     | YES   | fixed earlier (UPSTREAM.md) |
| rnd     | rnd             | --beta 1     | YES   | unmodified |
| icm     | icm             | --beta 1     | YES   | unmodified |
| ama     | ama             | --beta 1     | YES   | coeff fix (ext=1,int=beta) |
| ppo     | ppo             | --epsilon 0  | YES   | softmax/entropy baseline |
| egreedy | ppo             | --epsilon 0.1| YES   | epsilon-greedy baseline |

UCB / Thompson: discussion-only (no scalable deep-RL drop-in).
