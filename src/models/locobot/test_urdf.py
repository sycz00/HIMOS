import pybullet as p

p.connect(p.GUI)
p.loadURDF('locobot.urdf')

while True:
	p.stepSimulation()