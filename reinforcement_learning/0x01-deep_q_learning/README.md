# 0x01. Deep Q-learning

## Breakout:

My goal was to create a deep Q-learning model that could play breakout. This
model preforms well enough, but I could not get it to move fast enough on my
hardware. This type of training is very CPU intensive and I did not want to 
burn out my laptop. That being said I think it does pretty good, up until it
"breaks out" and then it gets confused and does not want to launch the ball,
from what I have read even google is now using an auto launch feature for this
particular problem. I also need to further tune the hyper parameters especially
gamma, and I would have liked to increase the frame capture up to one frame per
frame.
There are two main files, play.py and train.py, and I have chosen to include my
policy.h5 for myself. play.py loads the policy, after recreating the env that it
was trained on, and plays up to 10 games of breakout, however if a "breakout"
happens it gets confused. Rarely it will fail to launch altogether but this is a
common bug and just requires two things, either additional training, or a higher
epsilon, so it takes more risk. The train.py file trains the model and saves call
backs of the model every 100000 steps, that way if for some reason it does better
after only completing half of the training or the training is interrupted it is 
easy enough to load and continue. Once training is complete it will save the policy
as policy.h5