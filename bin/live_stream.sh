#!/bin/bash
#Tim Rappold
#June 18, 2018

url="https://stream-us1-alfa.dropcam.com/nexus_aac/1ccf6033fb9c41fd9f17cdf5716d318e/playlist.m3u8"

ffmpeg -i $url -c copy -map a -f segment -segment_time 5 -strftime 1 "../data/live_stream_conversions/audio_chunk_%Y-%m-%d_%H-%M-%S.wav"
