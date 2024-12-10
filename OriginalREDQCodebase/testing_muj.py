import mujoco_py
# print(mujoco_py.discover_mujoco())

model = mujoco_py.load_model_from_path("/home/romeo/.mujoco/mujoco210/model/hopper.xml")
print(model)


