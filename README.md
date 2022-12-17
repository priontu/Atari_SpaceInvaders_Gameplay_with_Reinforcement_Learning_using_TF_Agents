# Atari_SpaceInvaders_Gameplay_with_Reinforcement_Learning_using_TF_Agents

In this project, we implement Reinforcement Learning to develop an Agent that teaches itself to play the Atari SpaceInvaders Game. The project was developed using tensorflow, TF-Agents and OpenAI Gym on Google Cloud Platform (GCP).

Reinforcement learning (RL) is the area of machine learning concerned with intelligent agents that take actions in an environment in order to maximize the notion of cumulative reward.

For this project, we use the OpenAI Gym Atari environment, and we train an agent to play efficiently in this environment. The goal is to develop an agent that can exercise Human-Level Control.

The model consists of the following components:

**Agent** -- The agent is the decision-maker (bot) that interacts with the environment and learns the decisions that maximize the rewards.

**Environment** -- The environment is the space with which the agent interacts to obtain the rewards. 

**Observation space** -- This is the current state of the environment that the agent observes.

**Action space** -- This is the decision that the robot takes at each step.

**Reward** -- In this case, the score of the game, that the agent will try to maximize.

**Policy** -- A policy defines the learning agent's way of behaving at a given time. It is a mapping from perceived states of the environment to actions to be taken when in those states.

Agent's Gameplay before training (around 1000 iterations):

![space_invaders_2M_9](https://user-images.githubusercontent.com/61733487/208233183-ed32b5c8-6ee9-41b3-a306-068cf013e11d.gif)

At this point the agent doesn't know much. It's trying different things and learning what works, and which actions let it earn some rewards.

Agent's Gameplay mid-training (near 1.5M iterations):

![space_invaders_2M_14_1st_win](https://user-images.githubusercontent.com/61733487/208233316-d3359c27-0673-4306-bcff-b57815d50d62.gif)

The agent learned some new things -- now it knows it needs to shoot the enemy and kill them in order to gain points, but it hasn't yet learned how to stay alive by evading the enemy fire and kill all the enemies in a round.

Agent's Gameplay after training (over 3M iterations):

![space_invaders_26_2_wins](https://user-images.githubusercontent.com/61733487/208233550-21ec99f4-b590-4b6a-ba60-aad93608bf64.gif)

At this point in the training, we have reached Human-Level Control in the Gameplay.

We can see that the Gameplay has improved. The agent has learned how to shoot and kill the enemy in order to earn rewards, but it has also learned actions like following enemy movements and evading enemy gunfire which would allow it to earn greater rewards. However, it definitely requries some more training, since the enemy seems to have some more tricks up its sleeve, like increasing speed of movement and gunfire. The agent tends to adapt to this sometimes, which is promising. This allows the agent to win some rounds and jump to the next. However, it always fails in the second round.

These would be the same setbacks a human player would face, which he would improve on by playing the game some more. The same applies to our agent. 

References:

[1] https://www.nature.com/articles/nature14236

[2] Hands-On Machine Learning with Scikit-Learn and TensorFlow, by Aurélien Géron

