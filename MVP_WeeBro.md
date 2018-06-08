# WeeBro: A Baby Monitor that Tracks Crying 

AriaCrying?

Author: Tim Rappold

## Project Summary

**MVP**:

The minimum viable product is a classifier that  identfies a baby's (in particular, our baby's) cry from audio signals. The minimum feature will identify from an audio signal whether or not a crying baby can be heard.

**Additional Features**: A whole suite of functionality can be built on top of this classifier. Here's a list of addional features I'd like to attempt:

* **Integrate with NEST API**. We have a Nest camera in the baby's room. We use it mostly for security, facing out onto the street. However, it also has a microphone that picks up ambeint audio signals. An integration with the Nest API should allow me to stream audio data in real-time and pass it to the WeeBro classifier to create a *time-series of classifications*.
* **Build a live monitoring tool**. This tool would build on top of the NEST API interface by constantly reading and classifying the audio stream.
* **Flask app with dashboard**: This app would provide a (running) dashboard of cry/no-cry classification, as well as basic statistics for the last few hours/days/weeks, such as `number of minutes crying`in a given day, and/or the average for a given week.

**Domain:**  Classification, audio processing, linear machine learning and deep learning models.

**Data:** 

* **Existing audio collections on Github**: There are a number of audio libraries available from similar projects, including various degress of baby cries, user-submitted labeled data from boys and girls in various ages, and background noises.

  | Repo                                                     | Types of Labeled Data                                        | File Types |
  | -------------------------------------------------------- | ------------------------------------------------------------ | ---------- |
  | [mystesPF](https://github.com/mystesPF)                  | Diverse set of sound files including several samples each from the following categories: crying babies, snoring, glass breaking, dog barking, chirping birds, car horn, silence, noise, thunderstorm, etc. | *.ogg      |
  | [giulbia](https://github.com/giulbia/baby_cry_detection) |                                                              |            |
  |                                                          |                                                              |            |

  

* **Audio from our daughter**: extract nightly audio from nest cam video.



## Project Milestones

1.) Public Data validation (THURSDAY):

​	Download public data set(s)

​	Sample 20 sound clips

​	Demonstrate ability to open/process all included file formats



2.) Personal Data validation (FRIDAY):

​	Download and prep sound files from Nest [partially complete]

​	Demonstrate audio extraction from video file [complete, via ffmpeg]



3.) Validate Nest API (SUNDAY)

​	Demonstrate that I can access camera video live stream, figure out how this works.



4.) Demonstrate signal extraction and data formating (SUNDAY):

​	Extract signals from 1-3 audio files using e.g. Librosa

​	Set up training data table 



5.) Build a first linear model (MONDAY)

6.) Build a first live stream monitor (TUESDAY)

6.) Build a first CNN (WEDNESDAY)

7.) Build a first Flask app and Bokeh dashboard (THURSDAY)

8.) Wrap up first prototype (FRIDAY - SUNDAY)

9.) Refine (Week 2: MONDAY - SUNDAY)

10.) Presentation (Week 3)

 