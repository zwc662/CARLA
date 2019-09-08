'''
Script to replay a recorded file, showing log info including collision and block
Ref: https://carla.readthedocs.io/en/latest/recorder_and_playback/
Also see start_replaying for sample code.
'''

import os, sys
import numpy as np
import pygame

import carla
from synchronous_mode_client_control_test_NN import get_font

def replay(filepath):
	pygame.init()
	display = pygame.display.set_mode(
		(800, 600),
		pygame.HWSURFACE | pygame.DOUBLEBUF)
	font = get_font()
	clock = pygame.time.Clock()

	client = carla.Client('localhost', 2000)
	client.set_timeout(2.0)

	world = client.get_world()
	world.set_weather(carla.WeatherParameters.ClearNoon)
	blueprint_library = world.get_blueprint_library()

	# replay the record file
	client.replay_file(filepath, 0.0, 0.0, 0)
	# # for more options
	# client.replay_file(filepath, start, duration, camera) # unit: s
	client.set_replayer_time_factor(1.0)  # the pedestrian will not be affected
	# client.show_recorder_file_info(filepath)

	# # comment the following for more info of collision and blocking
	# client.show_recorder_collisions(filepath, "a", "a")
	# client.show_recorder_actors_blocked(filepath, min_time, min_distance)


if __name__ == '__main__':

	try:

		filepath = "/home/ruihan/.config/Epic/CarlaUE4/Saved/record_testing_dest_0_start_2.log"
		replay(filepath)

	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')