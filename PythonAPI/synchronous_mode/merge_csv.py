import os
from glob import glob
import csv

files = []
# start_dir = os.getcwd()
start_dir = '/home/ruihan/UnrealEngine_4.22/carla/Dist/CARLA_0.9.5-428-g0ce908db/LinuxNoEditor/PythonAPI/synchronous_mode/data/dest_start_SR0.5'
pattern   = "all*.csv"

for dir,_,_ in os.walk(start_dir):
	files.extend(glob(os.path.join(dir,pattern))) 

print("pattern", pattern)
# for file in files:
# 	print(file.split('/')[-1])

csv_out = '/home/ruihan/UnrealEngine_4.22/carla/Dist/CARLA_0.9.5-428-g0ce908db/LinuxNoEditor/PythonAPI/synchronous_mode/data/dest_start_SR0.5/merged.csv'
with open(csv_out, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(["throttle", "steer", \
                     "cur_speed", "cur_x", "cur_y", "cur_z", "cur_pitch", "cur_yaw", "cur_roll", \
                     "target_speed", "target_x", "target_y", "target_z", "target_pitch", "target_yaw", "target_roll"])

for file in files:
	print('append file', file.split('/')[-1])
	with open(file)as fin:
		csv_reader = csv.reader(fin)
		# line_count = sum(1 for line in csv_reader)
		# print("line count", line_count)
		for line in csv_reader:
			if line[0].startswith("throttle"):
				print("header")
				continue
			with open(csv_out, 'a+') as fout:
				csv_writer = csv.writer(fout)
				csv_writer.writerow(line)


csv_merge = open(csv_out, 'r')
total_row_count = sum(1 for row in csv_merge) 
print('Finish writing : ' + csv_out + " {} lines".format(total_row_count))