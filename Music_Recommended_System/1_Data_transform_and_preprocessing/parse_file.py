# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/15 14:24
@Project:Music_Recommended_System
@Filename:parse_file.py
"""


import json

def parse_song_line(in_line):
	data = json.loads(in_line)
	name = data['result']['name']
	tags = ",".join(data['result']['tags'])
	subscribed_count = data['result']['subscribedCount']
	if subscribed_count < 100:
		return False
	playlist_id = data['result']['id']
	song_info = ''
	songs = data['result']['tracks']
	for song in songs:
		try:
			song_info += "\t" + ":::".join((str(song['id']),song['name'],song['artists'][0]['name'],str(song['popularity'])))
		except Exception as e:
			print(e)
			continue
	return name + "##" + tags + "##" + str(playlist_id) + "##" + str(subscribed_count) + song_info

def parse_file(in_file,out_file):
	out_file = open(out_file,'w')
	for line in open(in_file):
		result = parse_song_line(line)
		if result:
			out_file.write(result.encode('utf-8').strip() + "\n")
	out_file.close()