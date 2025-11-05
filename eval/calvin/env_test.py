import os

# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"
import pyrender

from calvin_env.envs.play_table_env import get_env

path = "/home/disk1/jianke_z/task_ABC_D/validation"
env = get_env(path, show_gui=False)
# print(env.get_obs())
obs = env.get_obs()
rgb_static = obs['rgb_obs']['rgb_static']
rgb_gripper = obs['rgb_obs']['rgb_gripper']
import PIL.Image

print(obs)
PIL.Image.fromarray(rgb_static).save('rgb_static.png')
PIL.Image.fromarray(rgb_gripper).save('rgb_gripper.png')
