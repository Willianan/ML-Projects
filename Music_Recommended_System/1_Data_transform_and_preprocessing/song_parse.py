# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/15 14:55
@Project:Music_Recommended_System
@Filename:song_parse.py
"""


def is_null(s):
	return len(s.split(",")) > 2

def parse_song_info(song_info):
	try:
		song_id,name,artist,popularity = song_info.split(":::")
		return ",".join([song_id,"1.0","1300000"])
	except Exception as e:
		return ""

def parse_playlist_line(in_line):
	try:
		contents = in_line.strip().split("\t")
		name,tags,playlist_id,subscribed_count = contents[0].split("##")
		songs_info = map(lambda x:playlist_id + "," + parse_song_info(x),contents)
		songs_info = filter(is_null,songs_info)
		return "\n".join(songs_info)
	except Exception as e:
		print(e)
		return False

def parse_file(in_file,out_file):
	out_file = open(out_file,'w')
	for line in open(in_file):
		result = parse_playlist_line(line)
		if result:
			out_file.write(result.encode('utf-8').strip() + "\n")
	out_file.close()
