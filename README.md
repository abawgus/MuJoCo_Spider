# MuJoCo_Spider

Python 3.11 required.
See spec_file_gym_re for the list of packages for the conda environment required.


To run the MuJoCo spider, the spider files must be added to the Gym library.

(1) In the conda environment, go to '/Lib/site_packages/gymnasium/env/__init__.py'

Add the following text to the __init__ file:

register(
     id="Spider-v0",
     entry_point="gymnasium.envs.mujoco.spider:SpiderEnv",
     max_episode_steps=1000,
     reward_threshold=6000.0,
)

(2) To '/Lib/site_packages/gymnasium/env/mujoco', add the 'spider.py' file.

(3) To '/Lib/site_packages/gymnasium/env/mujoco/assets', add the 'spider.xml' file.

The example runtime file is located as PPO_Class.py.
