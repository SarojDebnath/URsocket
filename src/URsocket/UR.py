import os
import sys
directory=os.path.dirname(os.path.abspath(__file__))
sys.path.append(directory)

import subprocess
import time
import datetime
import logging
import rtde as rtde
import rtde_config as rtde_config
import socket
import numpy as np

class control:
    def __init__(self,ip):
        self.ip=ip
        #SETUP READ#
        self.readport=30004
        self.writeport=30002
        self.config_filename=f'{directory}/config.xml'
        #Initialize RTDE Connection
        logging.getLogger().setLevel(logging.INFO)
        self.conf = rtde_config.ConfigFile(self.config_filename)
        self.state_names, self.state_types = self.conf.get_recipe('state')
        self.watchdog_names, self.watchdog_types = self.conf.get_recipe('watchdog')
        self.con = rtde.RTDE(self.ip,self.readport)
        self.con.connect()
        connected = self.con.is_connected()
        if connected == True:
            print('Connected')
        else:
            print('Not able to connect')
            raise ConnectionError('Check Connection')

        self.con.get_controller_version()
        self.con.send_output_setup(self.state_names, self.state_types)
        self.watchdog = self.con.send_input_setup(self.watchdog_names, self.watchdog_types)
        self.watchdog.input_int_register_0 = 0
        
        ###WRITE###SETUP###
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.ip, 30002))

    def sendscript(self,cmd,hold):
        sim=False
        if hold==True and sim==False:
            sim=True
            newline=cmd.find('\n')
            cmd=cmd[:newline+1]+'  write_output_integer_register(24,0)\n'+cmd[newline+1:]
            cmd=cmd[:-4]+'  write_output_integer_register(24,1)\n'+cmd[-4:]
        self.s.send ((cmd).encode('utf-8'))
        while hold==True:
            if not self.con.send_start():
                sys.exit()
            state = self.con.receive()
            if state.output_int_register_24==1:
                self.s.send (('def unnamed():\n  write_output_integer_register(24,0)\nend\n').encode('utf-8'))
                self.con.send(self.watchdog)
                break
    def sendscriptfile(self,file):
        f_open = open (file, "rb")
        l_open = f_open.read(1024)
        while (l_open):
            self.s.send(l_open)
            l_open = f_open.read(1024)
    def get_value(self, value_name):
        if not self.con.send_start():
            sys.exit()
        state = self.con.receive()
        value = getattr(state, value_name)
        self.con.send(self.watchdog)
        return value
    def get_6dvector(self):
        if not self.con.send_start():
            sys.exit()
        state = self.con.receive()
        value = [getattr(state, f'output_double_register_{i}') for i in range(24, 30)]
        self.con.send(self.watchdog)
        return value
        
    def dashcomm(self,command):
        command=f'echo y | plink root@{self.ip} -pw easybot "{{ echo \\"{command}\\"; echo \\"quit\\"; }} | nc 127.0.0.1 29999"'
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(0.25)
        stdout, stderr = process.communicate()
        return stdout.decode().split('\n')[1:-2]
    
    def disconnect(self):
        self.con.disconnect()
        self.s.close()
        
    def endForceMood(self):
        cmd='def unnamed():\n  end_force_mode()\nend\n'
        self.sendscript(cmd,True)

    def endFreedrive(self):
        cmd='def unnamed():\n  end_freedrive_mode()\nend\n'
        self.sendscript(cmd,True)
    def force_mode(self,task_frame,selection_vector, wrench, types,limits):
        cmd=f'def unnamed():\n  force_mode({task_frame},{selection_vector},{wrench},{types},{limits})\nend\n'
        self.sendscript(cmd,True)
    def freedrive(self,freeAxes,feature):#NA
        cmd=f'def unnamed():\n  freedrive_mode({freeAxes},p{feature})\nend\n'
        self.sendscript(cmd,True)
    def getFreedriveStatus(self):
        cmd='def unnamed():\n  var_1=get_freedrive_status()\n  write_output_integer_register(24,var_1)\nend\n'
        self.sendscript(cmd,True)
        return self.get_value('output_int_register_24')
    def moveC(self,pose_via,pose_to,a,v,r,mode):
        cmd=f'def unnamed():\n  movec(p{pose_via},p{pose_to},{a},{v},{r},{mode})\nend\n'
        self.sendscript(cmd,True)
    def moveJ(self,q,a,v,t,r):
        cmd=f'def unnamed():\n  movej({q},{a},{v},{t},{r})\nend\n'
        self.sendscript(cmd,True)
    def moveL(self,pose,a,v,t,r):
        cmd=f'def unnamed():\n  movel(p{pose},{a},{v},{t},{r})\nend\n'
        self.sendscript(cmd,True)
    def moveP(self,pose,a,v,r):
        cmd=f'def unnamed():\n  movep(p{pose},{a},{v},{r})\nend\n'
        self.sendscript(cmd,True)
    def servoC(self,pose,a,v,r):
        cmd=f'def unnamed():\n  servoc({pose},{a},{v},{r})\nend\n'
        self.sendscript(cmd,True)
    def servoJ(self,q,a,v,t,l,g):
        cmd=f'def unnamed():\n  servoj({q},{a},{v},{t},{l},{g})\nend\n'
        self.sendscript(cmd,True)
    def speedJ(self,qd,a,t,wait):
        cmd=f'def unnamed():\n  speedj({qd},{a},{t})\nend\n'
        self.sendscript(cmd,wait)
    def speedL(self,xd,a,t,aRot,wait):
        cmd=f'def unnamed():\n  speedl({xd},{a},{t},{aRot})\nend\n'
        self.sendscript(cmd,wait)
        
    def moveUntillcontact(self,d,a):
        cmd=cmd=f'def moveToolContact():\n  while True:\n    step_back = tool_contact()\n    if step_back <= 0:\n      speedl({d}, {a}, t=get_steptime())\n    else:\n      stopl(3)\n      break\n    end\n  end\nend\n'
        self.sendscript(cmd,True)
        
    def movelToolspace(self,pose,wait,a,v,t=0,r=0.0):
        cmd=f'def unnamed():\n  movel(pose_trans(get_actual_tcp_pose(),p{pose}),{a},{v},{t},{r})\nend\n'
        self.sendscript(cmd,wait)
    def stopL(self,a):
        cmd=f'def unnamed():\n  stopl({a})\nend\n'
        self.sendscript(cmd,True)
    def stopJ(self,a):
        cmd=f'def unnamed():\n  stopj({a})\nend\n'
        self.sendscript(cmd,True)
    def getForce(self): return self.get_value('tcp_force_scalar')

    def getActualJointPositions(self,rad=True):
        value=self.get_value('actual_q')
        if rad==False:
            value=list(map(np.rad2deg,value))
        return value

    def getActualJointSpeeds(self):
        cmd='def unnamed():\n  global var_1=get_actual_joint_speeds()\n  write_output_float_register(24,var_1[0])\n  write_output_float_register(25,var_1[1])\n  write_output_float_register(26,var_1[2])\n  write_output_float_register(27,var_1[3])\n  write_output_float_register(28,var_1[4])\n  write_output_float_register(29,var_1[5])\nend\n'
        self.sendscript(cmd,True)
        return self.get_6dvector()
    
    def getActualTCPPose(self): return self.get_value('actual_TCP_pose')
        
    def getActualTCPSpeed(self): return self.get_value('actual_TCP_speed')

    def getControllerTemp(self):
        cmd='def unnamed():\n  var_1=get_controller_temp()\n  write_output_float_register(30,var_1)\nend\n'
        self.sendscript(cmd,True)
        return self.get_value('output_double_register_30')
    def getForwardKin(self,q,tcp):
        cmd=f'def unnamed():\n  global var_1=get_forward_kin({q},p{tcp})\n  write_output_float_register(24,var_1[0])\n  write_output_float_register(25,var_1[1])\n  write_output_float_register(26,var_1[2])\n  write_output_float_register(27,var_1[3])\n  write_output_float_register(28,var_1[4])\n  write_output_float_register(29,var_1[5])\nend\n'
        self.sendscript(cmd,True)
        return self.get_6dvector()
    
    def getInverseKin(self,x,qnear,maxError,maxOrientationError,tcp):
        
        cmd=f'def unnamed():\n  global var_1=get_inverse_kin(p{x},{qnear},{maxError},{maxOrientationError},p{tcp})\n  write_output_float_register(24,var_1[0])\n  write_output_float_register(25,var_1[1])\n  write_output_float_register(26,var_1[2])\n  write_output_float_register(27,var_1[3])\n  write_output_float_register(28,var_1[4])\n  write_output_float_register(29,var_1[5])\nend\n'
        self.sendscript(cmd,True)
        return self.get_6dvector()
    def getInverseKinSol(self,x,qnear,maxError,maxOrientationError,tcp):
        cmd=f'def unnamed():\n  global var_1=get_inverse_kin_has_solution(p{x},{qnear},{maxError},{maxOrientationError},p{tcp})\n  write_output_boolean_register(64,var_1)\nend\n'
        self.sendscript(cmd,True)
        return self.get_value('output_bit_register_64')
    def getJointTemp(self): return self.get_value('joint_temperatures')
        
    def getJointTorque(self):
        cmd='def unnamed():\n  global var_1=get_joint_torques()\n  write_output_float_register(24,var_1[0])\n  write_output_float_register(25,var_1[1])\n  write_output_float_register(26,var_1[2])\n  write_output_float_register(27,var_1[3])\n  write_output_float_register(28,var_1[4])\n  write_output_float_register(29,var_1[5])\nend\n'
        self.sendscript(cmd,True)
        return self.get_6dvector()
    def getTargetJoint(self): return self.get_value('target_q')
        
    def getTargetJointSpeed(self): return self.get_value('target_qd')
    def getTargetJointAccel(self): return self.get_value('target_qdd')
    def getPayload(self): return self.get_value('payload')
    def getPayloadCG(self): return self.get_value('payload_cog')
    def getPayloadInertia(self): return self.get_value('payload_inertia')
    
    def getTCPForce(self): return self.get_value('actual_TCP_force')
        
    def getTCPOffset(self):
        cmd='def unnamed():\n  global var_1=get_tcp_offset()\n  write_output_float_register(24,var_1[0])\n  write_output_float_register(25,var_1[1])\n  write_output_float_register(26,var_1[2])\n  write_output_float_register(27,var_1[3])\n  write_output_float_register(28,var_1[4])\n  write_output_float_register(29,var_1[5])\nend\n'
        self.sendscript(cmd,True)
        return self.get_6dvector()
    def getToolCurrent(self): return self.get_value('tool_output_current')
    def isSteady(self):
        cmd='def unnamed():\n  global var_1=is_steady()\n  write_output_boolean_register(64,var_1)\nend\n'
        self.sendscript(cmd,True)
        return self.get_value('output_bit_register_64')
    def isSafe(self,pose):
        cmd=f'def unnamed():\n  global var_1=is_within_safety_limits({pose})\n  write_output_boolean_register(64,var_1)\nend\n'
        self.sendscript(cmd,True)
        return self.get_value('output_bit_register_64')
    def setGravity(self,d):
        cmd=f'def unnamed():\n  set_gravity({d})\nend\n'
        self.sendscript(cmd,True)
    def setPayload(self,m,CG):
        cmd=f'def unnamed():\n  set_payload({m}, {CG})\nend\n'
        self.sendscript(cmd,True)
        return self.getPayload(),self.getPayloadCG()
    def setPayloadCG(self,CG):
        cmd=f'def unnamed():\n  set_payload_cog({CG})\nend\n'
        self.sendscript(cmd,True)
        return self.getPayloadCG()
    def setPayloadMass(self,m):
        cmd=f'def unnamed():\n  set_payload_mass({m})\nend\n'
        self.sendscript(cmd,True)
        return self.getPayload()
    def setTargetPayload(self,m,CG,inertia):
        cmd=f'def unnamed():\n  set_target_payload({m},{CG},{inertia})\nend\n'
        self.sendscript(cmd,True)
        return self.getPayload(),self.getPayloadCG(),self.getPayloadInertia()
    def setTCP(self,pose):
        cmd=f'def unnamed():\n  set_tcp(p{pose})\nend\n'
        self.sendscript(cmd,True)
        return self.getTCPOffset()
    #def toolContact(self):
        #pass
    def interpolatepose(self,p_from,p_to,alpha):
        cmd=f'def unnamed():\n  global var_1=interpolate_pose(p{p_from}, p{p_to}, {alpha})\n  write_output_float_register(24,var_1[0])\n  write_output_float_register(25,var_1[1])\n  write_output_float_register(26,var_1[2])\n  write_output_float_register(27,var_1[3])\n  write_output_float_register(28,var_1[4])\n  write_output_float_register(29,var_1[5])\nend\n'
        self.sendscript(cmd,True)
        return self.get_6dvector()
    def pointdist(self,p_from,p_to):
        cmd=f'def unnamed():\n  var_1=point_dist(p{p_from}, p{p_to})\n  write_output_float_register(30,var_1)\nend\n'
        self.sendscript(cmd,True)
        return self.get_value('output_double_register_30')
    def posedist(self,pfrom,pto):
        cmd=f'def unnamed():\n  var_1=pose_dist(p{pfrom}, p{pto})\n  write_output_float_register(30,var_1)\nend\n'
        self.sendscript(cmd,True)
        return self.get_value('output_double_register_30')
    def poseInv(self,p_from):
        cmd=f'def unnamed():\n  global var_1=pose_inv(p{p_from})\n  write_output_float_register(24,var_1[0])\n  write_output_float_register(25,var_1[1])\n  write_output_float_register(26,var_1[2])\n  write_output_float_register(27,var_1[3])\n  write_output_float_register(28,var_1[4])\n  write_output_float_register(29,var_1[5])\nend\n'
        self.sendscript(cmd,True)
        return self.get_6dvector()
    def poseTrans(self,p_from,p_to):
        cmd=f'def unnamed():\n  global var_1= pose_trans(p{p_from}, p{p_to})\n  write_output_float_register(24,var_1[0])\n  write_output_float_register(25,var_1[1])\n  write_output_float_register(26,var_1[2])\n  write_output_float_register(27,var_1[3])\n  write_output_float_register(28,var_1[4])\n  write_output_float_register(29,var_1[5])\nend\n'
        self.sendscript(cmd,True)
        return self.get_6dvector()
    def rotTorpy(self,rVec):
        cmd=f'def unnamed():\n  global var_1= rotvec2rpy({rVec})\n  write_output_float_register(24,var_1[0])\n  write_output_float_register(25,var_1[1])\n  write_output_float_register(26,var_1[2])\nend\n'
        self.sendscript(cmd,True)
        return [self.get_value('output_double_register_24'),self.get_value('output_double_register_25'),self.get_value('output_double_register_26')]
    def rpyTorot(self,rpy_vector):
        cmd=f'def unnamed():\n  global var_1= rpy2rotvec({rpy_vector})\n  write_output_float_register(24,var_1[0])\n  write_output_float_register(25,var_1[1])\n  write_output_float_register(26,var_1[2])\nend\n'
        self.sendscript(cmd,True)
        return [self.get_value('output_double_register_24'),self.get_value('output_double_register_25'),self.get_value('output_double_register_26')]
    def wrenchTrans(self,Tfrom,Wfrom):
        cmd=f'def unnamed():\n  global var_1= wrench_trans(p{Tfrom}, {Wfrom})\n  write_output_float_register(24,var_1[0])\n  write_output_float_register(25,var_1[1])\n  write_output_float_register(26,var_1[2])\n  write_output_float_register(27,var_1[3])\n  write_output_float_register(28,var_1[4])\n  write_output_float_register(29,var_1[5])\nend\n'
        self.sendscript(cmd,True)
        return self.get_6dvector()
    def getAnalogIOTypes(self): return self.get_value('analog_io_types')
    def getAllAnalogIn(self): return [self.get_value('standard_analog_input0'),self.get_value('standard_analog_input1')]
    def getAnalogIn(self,n): return self.get_value(f'standard_analog_input{n}')
        
    def getAllAnalogOut(self): return [self.get_value('standard_analog_output0'),self.get_value('standard_analog_output1')]
        
    def getAnalogOut(self,n): return self.get_value(f'standard_analog_output{n}')
        
    def getConfigurableDigitalIn(self,n):
        allIn=self.getAllDigitalIn()
        configIn=allIn[8:16]
        return configIn[n]
    def getConfigurableDigitalOut(self,n):
        allOut=self.getAllDigitalOut()
        configOut=allOut[8:16]
        return configOut[n]
    def getAllDigitalIn(self):
        value=self.get_value('actual_digital_input_bits')
        binary_string = bin(value)[2:]
        binary_list = [0] * (18 - len(binary_string)) + [int(digit) for digit in binary_string]
        boolean_list = [bool(digit) for digit in binary_list][::-1]
        return boolean_list
    def getDigitalIn(self,n):
        allIn=self.getAllDigitalIn()
        return allIn[n]
    def getAllDigitalOut(self): 
        value=self.get_value('actual_digital_output_bits')
        binary_string = bin(value)[2:]
        binary_list = [0] * (18 - len(binary_string)) + [int(digit) for digit in binary_string]
        boolean_list = [bool(digit) for digit in binary_list][::-1]
        return boolean_list
    def getDigitalOut(self,n):
        allOut=self.getAllDigitalOut()
        return allOut[n]
    def getStandardDigitalIn(self,n):
        allIn=self.getAllDigitalIn()
        standardIn=allIn[0:8]
        return standardIn[n]
    def getStandardDigitalOut(self,n):
        allOut=self.getAllDigitalOut()
        stdOut=allOut[0:8]
        return stdOut[n]
    def getToolAnlogInputTypes(self): return self.get_value('tool_analog_input_types')
    def getToolAnalogIn(self,n): return self.get_value(f'tool_analog_input{n}')
    def getToolDigitalIn(self,n):
        allIn=self.getAllDigitalIn()
        toolIn=allIn[16:18]
        return toolIn[n]
    def getToolDigitalOut(self,n):
        allOut=self.getAllDigitalOut()
        toolOut=allOut[16:18]
        return toolOut[n]
    def getToolOutVoltage(self): return self.get_value('tool_output_voltage')
    def setAnalogOutDomain(self,port,domain):
        cmd=f'def unnamed():\n  set_analog_outputdomain({port},{domain})\nend\n'
        self.sendscript(cmd,True)
    def setAnalogOut(self,n,f):
        cmd=f'def unnamed():\n  set_analog_out({n},{f})\nend\n'
        self.sendscript(cmd,True)
        return self.getAnalogOut(n)
    def setConfigurableDigitalOut(self,n,b):
        cmd=f'def unnamed():\n  set_configurable_digital_out({n},{b})\nend\n'
        self.sendscript(cmd,True)
        return self.getConfigurableDigitalOut(n)
    def setDigitalOut(self,n,b):
        cmd=f'def unnamed():\n  set_digital_out({n},{b})\nend\n'
        self.sendscript(cmd,True)
        return self.getDigitalOut(n)
    def setStandardAnalogOut(self,n,f):
        cmd=f'def unnamed():\n  set_standard_analog_out({n},{f})\nend\n'
        self.sendscript(cmd,True)
    def setStandardDigitalOut(self,n,b):
        cmd=f'def unnamed():\n  set_standard_digital_out({n},{b})\nend\n'
        self.sendscript(cmd,True)
        return self.getStandardDigitalOut(n)
    def setToolOutput(self,n):
        cmd=f'def unnamed():\n  set_tool_output_mode({n})\nend\n'
        self.sendscript(cmd,True)
    def setToolDigitalOutMode(self,n,mode):
        cmd=f'def unnamed():\n  set_tool_digital_output_mode({n},{mode})\nend\n'
        self.sendscript(cmd,True)
    def setToolDigitalOut(self,n,b):
        cmd=f'def unnamed():\n  set_tool_digital_out({n},{b})\nend\n'
        self.sendscript(cmd,True)
        return self.getToolDigitalOut(n)
    def setToolVoltage(self,voltage):
        cmd=f'def unnamed():\n  set_tool_voltage({voltage})\nend\n'
        self.sendscript(cmd,True)
        return self.getToolOutVoltage()
    
    def loadurp(self,filepath): return self.dashcomm(f'load {filepath}')
    def play(self): return self.dashcomm('play')
    def stopProgram(self): return self.dashcomm('stop')
    def pauseProgram(self): return self.dashcomm('pause')
    def shutdown(self):  return self.dashcomm('shutdown')
    def isProgramRunning(self): return self.dashcomm('running')
    def robotmode(self): return self.dashcomm('robotmode')
    def popupDash(self,text): return self.dashcomm(f'popup {text}')
    def closepopup(self): return self.dashcomm('close popup')
    def polyscopeversion(self): return self.dashcomm('PolyscopeVersion')
    def programState(self): return self.dashcomm('programState')
    def softwareversion(self): return self.dashcomm('version')
    def setOperationalMode(self,mode):
        self.dashcomm(f'set operational mode {mode}')
        self.clearOperationalMode()
        return self.getOperationalMode()
    def getOperationalMode(self): return self.dashcomm('get operational mode')
    def clearOperationalMode(self): return self.dashcomm('clear operational mode')
    def poweron(self): return self.dashcomm('power on')
    def poweroff(self): return self.dashcomm('power off')
    def brakerelease(self): return self.dashcomm('brake release')
    def safetystatus(self): return self.dashcomm('safetystatus')
    def unlockprotectivestop(self): return self.dashcomm('unlock protective stop')
    def closesafetypopup(self): return self.dashcomm('close safety popup')
    def loadinstallation(self,name): return self.dashcomm(f'load installation {name}')
    def restartsafety(self): return self.dashcomm('restart safety')
    def isinremote(self): return self.dashcomm('is in remote control')
    def getrobotSN(self): return self.dashcomm('get serial number')
    def getrobotModel(self): return self.dashcomm('get robot model')

    ###Robotiq###Gripper_Control