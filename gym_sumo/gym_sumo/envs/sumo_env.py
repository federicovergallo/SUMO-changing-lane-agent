import gym
import traci
import sumolib
import numpy as np
from scipy.spatial import distance
from math import atan2, degrees
from collections import deque
import os


def angle_between(p1, p2, rl_angle):
	xDiff = p2[0] - p1[0]
	yDiff = p2[1] - p1[1]
	angle = degrees(atan2(yDiff, xDiff))
	# Adding the rotation angle of the agent
	angle += rl_angle
	angle = angle % 360
	return angle


def get_distance(a, b):
	return distance.euclidean(a, b)


class SumoEnv(gym.Env):
	def __init__(self):
		self.name = 'rlagent'
		self.step_length = 0.4
		self.acc_history = deque([0, 0], maxlen=2)
		self.grid_state_dim = 3
		self.state_dim = (4*self.grid_state_dim*self.grid_state_dim)+1 # 5 info for the agent, 4 for everybody else
		self.pos = (0, 0)
		self.curr_lane = ''
		self.curr_sublane = -1
		self.target_speed = 0
		self.speed = 0
		self.lat_speed = 0
		self.acc = 0
		self.angle = 0
		self.gui = False
		self.numVehicles = 0
		self.vType = 0
		self.lane_ids = []
		self.max_steps = 10000
		self.curr_step = 0
		self.collision = False
		self.done = False


	def start(self, gui=False, numVehicles=30, vType='human', network_conf="networks/highway/sumoconfig.sumo.cfg", network_xml='networks/highway/highway.net.xml'):
		self.gui = gui
		self.numVehicles = numVehicles
		self.vType = vType
		self.network_conf = network_conf
		self.net = sumolib.net.readNet(network_xml)
		self.curr_step = 0
		self.collision = False
		self.done = False

		# Starting sumo
		home = os.getenv("HOME")

		if self.gui:
			sumoBinary = home + "/gitprograms/sumo/bin/sumo-gui"
		else:
			sumoBinary = home + "/gitprograms/sumo/bin/sumo"
		sumoCmd = [sumoBinary, "-c", self.network_conf, "--no-step-log", "true", "-W"]
		traci.start(sumoCmd)

		self.lane_ids = traci.lane.getIDList()

		# Populating the highway
		for i in range(self.numVehicles):
			veh_name = 'vehicle_' + str(i)
			traci.vehicle.add(veh_name, routeID='route_0', typeID=self.vType, departLane='random')
			# Lane change model comes from bit set 100010101010
			# Go here to find out what does it mean
			# https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html#lane_change_mode_0xb6
			#lane_change_model = np.int('100010001010', 2)
			lane_change_model = 256
			traci.vehicle.setLaneChangeMode(veh_name, lane_change_model)
		traci.vehicle.add(self.name, routeID='route_0', typeID='rl')

		# Do some random step to distribute the vehicles
		for step in range(self.numVehicles*4):
			traci.simulationStep()

		# Setting the lane change mode to 0 meaning that we disable any autonomous lane change and collision avoidance
		traci.vehicle.setLaneChangeMode(self.name, 0)

		# Setting up useful parameters
		self.update_params()


	def update_params(self):
		# initialize params
		self.pos = traci.vehicle.getPosition(self.name)
		self.curr_lane = traci.vehicle.getLaneID(self.name)
		if self.curr_lane == '':
			'''
			if we had collission, the agent is being teleported somewhere else. 
			Therefore I will do simulation step until he get teleported back
			'''
			assert self.collision
			while self.name in traci.simulation.getStartingTeleportIDList() or traci.vehicle.getLaneID(self.name) == '':
				traci.simulationStep()
			self.curr_lane = traci.vehicle.getLaneID(self.name)
		self.curr_sublane = int(self.curr_lane.split("_")[1])

		self.target_speed = traci.vehicle.getAllowedSpeed(self.name)
		self.speed = traci.vehicle.getSpeed(self.name)
		self.lat_speed = traci.vehicle.getLateralSpeed(self.name)
		self.acc = traci.vehicle.getAcceleration(self.name)
		self.acc_history.append(self.acc)
		self.angle = traci.vehicle.getAngle(self.name)


	# Get grid like state
	def get_grid_state(self, threshold_distance=10):
		'''
		Observation is a grid occupancy grid
		'''
		agent_lane = self.curr_lane
		agent_pos = self.pos
		edge = self.curr_lane.split("_")[0]
		agent_lane_index = self.curr_sublane
		lanes = [lane for lane in self.lane_ids if edge in lane]
		state = np.zeros([self.grid_state_dim, self.grid_state_dim])
		# Putting agent
		agent_x, agent_y = 1, agent_lane_index
		state[agent_x, agent_y] = -1
		# Put other vehicles
		for lane in lanes:
			# Get vehicles in the lane
			vehicles = traci.lane.getLastStepVehicleIDs(lane)
			veh_lane = int(lane.split("_")[-1])
			for vehicle in vehicles:
				if vehicle == self.name:
					continue
				# Get angle wrt rlagent
				veh_pos = traci.vehicle.getPosition(vehicle)
				# If too far, continue
				if get_distance(agent_pos, veh_pos) > threshold_distance:
					continue
				rl_angle = traci.vehicle.getAngle(self.name)
				veh_id = vehicle.split("_")[1]
				angle = angle_between(agent_pos, veh_pos, rl_angle)
				# Putting on the right
				if angle > 337.5 or angle < 22.5:
					state[agent_x, veh_lane] = veh_id
				# Putting on the right north
				if angle >= 22.5 and angle < 67.5:
					state[agent_x-1,veh_lane] = veh_id
				# Putting on north
				if angle >= 67.5 and angle < 112.5:
					state[agent_x-1, veh_lane] = veh_id
				# Putting on the left north
				if angle >= 112.5 and angle < 157.5:
					state[agent_x-1, veh_lane] = veh_id
				# Putting on the left
				if angle >= 157.5 and angle < 202.5:
					state[agent_x, veh_lane] = veh_id
				# Putting on the left south
				if angle >= 202.5 and angle < 237.5:
					state[agent_x+1, veh_lane] = veh_id
				if angle >= 237.5 and angle < 292.5:
					# Putting on the south
					state[agent_x+1, veh_lane] = veh_id
				# Putting on the right south
				if angle >= 292.5 and angle < 337.5:
					state[agent_x+1, veh_lane] = veh_id
		# Since the 0 lane is the right most one, flip 
		state = np.fliplr(state)
		return state
		
	def compute_jerk(self):
		return (self.acc_history[1] - self.acc_history[0])/self.step_length

	def detect_collision(self):
		collisions = traci.simulation.getCollidingVehiclesIDList()
		if self.name in collisions:
			self.collision = True
			return True
		self.collision = False
		return False
	
	def get_state(self):
		'''
		Define a state as a vector of vehicles information
		'''
		state = np.zeros(self.state_dim)
		before = 0
		grid_state = self.get_grid_state().flatten()
		for num, vehicle in enumerate(grid_state):
			if vehicle == 0:
				continue
			if vehicle == -1:
				vehicle_name = self.name
				before = 1
			else:
				vehicle_name = 'vehicle_'+(str(int(vehicle)))
			veh_info = self.get_vehicle_info(vehicle_name)
			idx_init = num*4
			if before and vehicle != -1:
				idx_init += 1
			idx_fin = idx_init + veh_info.shape[0]
			state[idx_init:idx_fin] = veh_info
		state = np.squeeze(state)
		return state
	
	
	def get_vehicle_info(self, vehicle_name):
		'''
			Method to populate the vector information of a vehicle
		'''
		if vehicle_name == self.name:
			return np.array([self.pos[0], self.pos[1], self.speed, self.lat_speed, self.acc])
		else:
			lat_pos, long_pos = traci.vehicle.getPosition(vehicle_name)
			long_speed = traci.vehicle.getSpeed(vehicle_name)
			acc = traci.vehicle.getAcceleration(vehicle_name)
			dist = get_distance(self.pos, (lat_pos, long_pos))
			return np.array([dist, long_speed, acc, lat_pos])
		
		
	def compute_reward(self, collision):
		'''
			Reward function is made of three elements:
			 - Comfort 
			 - Efficiency
			 - Safety
			 Taken from Ye et al.
		'''
		# Rewards Parameters
		alpha_comf = 0.5
		w_lane = 2.5
		w_speed = 2.5
		
		# Comfort reward 
		jerk = self.compute_jerk()
		R_comf = -alpha_comf*jerk**2
		
		#Efficiency reward
		try:
			lane_width = traci.lane.getWidth(traci.vehicle.getLaneID(self.name))
		except:
			print(traci.vehicle.getLaneID(self.name))
			lane_width = 3.2
		desired_x = self.pos[0] + lane_width*np.cos(self.angle)
		desired_y = self.pos[1] + lane_width*np.sin(self.angle)
		R_lane = -(np.abs(self.pos[0] - desired_x) + np.abs(self.pos[1] - desired_y))
		# Speed
		R_speed = -np.abs(self.speed - self.target_speed)
		# Eff
		R_eff = w_lane*R_lane + w_speed*R_speed
		
		# Safety Reward
		# Just penalize collision for now
		if collision:
			R_safe = -100
		else:
			R_safe = +1
		
		# total reward
		R_tot = R_comf + R_eff + R_safe
		return [R_tot, R_comf, R_eff, R_safe]
		
		
	def step(self, action):
		'''
		This will :
		- send action, namely change lane or stay 
		- do a simulation step
		- compute reward
		- update agent params 
		- compute nextstate
		- return nextstate, reward and done
		'''
		# Action legend : 0 stay, 1 change to right, 2 change to left
		if self.curr_lane[0] == 'e':
			action = 0
		if action != 0:
			if action == 1:
				if self.curr_sublane == 1:
					traci.vehicle.changeLane(self.name, 0, 0.1)
				elif self.curr_sublane == 2:
					traci.vehicle.changeLane(self.name, 1, 0.1)
			if action == 2:
				if self.curr_sublane == 0:
					traci.vehicle.changeLane(self.name, 1, 0.1)
				elif self.curr_sublane == 1:
					traci.vehicle.changeLane(self.name, 2, 0.1)
		# Sim step
		traci.simulationStep()
		# Check collision
		collision = self.detect_collision()
		# Compute Reward 
		reward = self.compute_reward(collision)
		# Update agent params 
		self.update_params()
		# State 
		next_state = self.get_state()
		# Update curr state
		self.curr_step += 1
		'''
		if self.curr_step > self.max_steps:
			done = True
			self.curr_step = 0
		else:
			done = False
		'''
		# Return
		done = collision
		return next_state, reward, done, collision
		
	def render(self, mode='human', close=False):
		pass

	def reset(self, gui=False, numVehicles=20, vType='human'):
		self.start(gui, numVehicles, vType)
		return self.get_state()

	def close(self):
		traci.close(False)
