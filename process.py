import os



ALL_txt = os.listdir('data/office_home')


for i in ALL_txt:
	old_file = os.path.join('data/office_home', i)
	new_file = os.path.join('data/office_home_', i)
	f = open(old_file, 'r')
	new_f = open(new_file, 'w')
	for line in f.readlines():
		new_line = line[60:]
		new_f.write(new_line)

	f.close()
	new_f.close()