import UR
robot=UR.control('172.25.121.84')
q=robot.getActualJointPositions(rad=True)
print(q)
p=robot.getActualTCPPose()
print(robot.setTCP([0.1,0.2,0,0,0,0]))
#print(robot.getJointTorque())
k=robot.getAllDigitalOut()[16:18]
print(k)
print(robot.getAllDigitalOut())
robot.disconnect()