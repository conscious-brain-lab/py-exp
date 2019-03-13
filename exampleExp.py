""""
Example experiment, to show how to use exptools in combination with PsychoPy. 
Visual detection task.

Created by Stijn Nuiten on 2019-03-13.
Copyright (c) 2019; All rights reserved.
"""

import os, sys, datetime
import subprocess
import pickle, datetime, time
import numpy as np
from math import *
from IPython import embed as shell
import shutil
from psychopy import logging, visual, clock, event, core
from psychopy import parallel
from psychopy.sound import Sound

logging.console.setLevel(logging.CRITICAL)

import exptools
from exptools.core.trial import Trial
from exptools.core.session import EyelinkSession

from psychopy.tools.attributetools import attributeSetter, setAttribute
from psychopy.visual import GratingStim, TextStim, ImageStim, NoiseStim, DotStim, Window

from pygaze import libscreen
from pygaze import libtime
from pygaze import liblog
from pygaze import libinput
from pygaze import eyetracker

fullscr = True 			
tracker_on = False 			# Set to True for eye-tracking
use_parallel = False		# Set to True for EEG
NR_TRIALS = 20				
block_length = 10
if use_parallel:
	from ctypes import windll
	portaddress = 0xA0D0
	port = windll.inpout32
	
	
class DetectTrial(Trial):
	def __init__(self, parameters = {}, phase_durations=[], session=None, screen=None, tracker=None, ID=0): # self, parameters = {}, phase_durations=[], session=None, screen=None, tracker=None, ID=0,
		self.screen = screen
		self.parameters = parameters
		self.ID = ID
		self.phase_durations = phase_durations  
		self.session = session
		self.block = np.floor(self.ID/self.session.block_length)
		self.tracker = tracker
		self.create_stimuli()

		if self.ID == 0:
			self.session_start_time = clock.getTime()

		self.run_time = self.prestimulation_time = self.delay_1_time = self.delay_2_time = self.answer_time = self.stimulus_time = 0.0
		self.parameters.update({'answer': -1, 
								'correct': -1,
								'block': self.block,
								'restarted': 0,
								})
		
		self.p_v_feedback_played =  self.p_stim_played = self.fixLost = False

		self.stopped = False
		super(
			DetectTrial,
			self).__init__(
			phase_durations=phase_durations,
			parameters = parameters,
			screen = self.screen,
			session = self.session,
			tracker = self.tracker			
			)

	def create_stimuli(self):

		self.center = ( self.screen.size[0]/2.0, self.screen.size[1]/2.0 )
		self.fixation = GratingStim(self.screen, tex='sin', mask = 'circle',size=4, pos=[0,0], sf=0, color ='white')
		self.fix_bounds = np.array([self.center + self.session.pixels_per_degree, self.center - self.session.pixels_per_degree])
		
		# The stimulus should covers 24-degree of visual angle 
		stim_size = 24 * round(self.session.pixels_per_degree) #round(10 * self.session.pixels_per_degree)
		noise_size = 27 * round(self.session.pixels_per_degree)
		spatial_freq = self.parameters['spatial_freq']

		self.grating = GratingStim(self.screen,contrast = 1,opacity= 1,  tex = "sin", mask="gauss", units='pix',  size=stim_size, sf = spatial_freq, color=[1,1,1]) 
		self.noise = NoiseStim(self.screen, contrast=0.5, name='noise', 
			units='pix', mask='raisedCos', size=(noise_size,noise_size), 
			opacity= 0.5, blendmode='avg', noiseType='binary', noiseElementSize=2)
	
	
		orientation = np.random.randint(0,2)
		self.parameters['orientation'] = orientation
		self.grating.ori = -45 + self.parameters['orientation'] * 90

		if self.ID % self.session.block_length == 0 and self.ID > 0:
			perf = np.array(self.session.corrects)[-self.session.block_length:][np.array(self.session.corrects)[-self.session.block_length:] >= 0].sum() / float(self.session.block_length) * 100.0
			intro_text = """\n%i%% goed! \n Druk op de spatiebalk om verder te gaan.""" % perf
			print(perf)
		else:
			intro_text = """In dit experiment is het de bedoeling dat je de aanwezigheid van een visueel patroon detecteert in ruis. Druk op toets 'K' als je denkt dat er een patroon te zien was in de ruis, of op toets 'S' als je denkt dat er geen patroon in de ruis zat. Je mag pas drukken als er geen stimulus meer op het scherm is! Probeer zo goed mogelijk op te letten om zoveel mogelijk goede antwoorden te kunnen geven! Druk op de spatiebalk om verder te gaan. \n Een voorbeeld van een patroon ziet er zo uit: """
				
		self.target_grating = GratingStim(self.screen,  tex = "sin", pos=(0,-300),mask="gauss", units='pix',  size=200, sf = 0.05, contrast = 1,ori = self.grating.ori, color=[1,1,1])                          
		self.message = visual.TextStim(self.screen, pos=[0,0],text= intro_text, color = (1.0, 1.0, 1.0), height=20)
		self.feedback = TextStim(self.screen, text = 'TOO SLOW', pos = (0,0 ), color = (1.0, 1.0, 1.0), height=40)

	def draw(self):
		# draw additional stimuli:
		if self.phase == 0: # 
			if self.ID == 0 or self.ID % self.session.block_length == 0:
				self.message.draw()
				self.target_grating.draw()
			else:
				self.fixation.draw()

		if self.phase == 1: # pre-stim cue
			self.fixation.color = 'black'
			self.fixation.draw()
		
		if self.phase ==2:  # Stimulus presentation
			self.fixation.draw()
			if self.parameters['signal_present'] == 1 :
				self.grating.draw()
			self.noise.draw()
			self.fixation.draw()
			
		elif self.phase == 3:# decision interval
			self.fixation.draw()

		elif self.phase == 4: # feedback
			if self.fixLost == True:
				self.fixation.color = 'red'
				self.fixation.draw()
			else:
				if self.parameters['answer'] == -1:
					self.feedback.draw()
				else:
					self.fixation.color = 'black'
					self.fixation.draw()
		super(DetectTrial, self).draw()

	def event(self):
		trigger = None
		for ev in event.getKeys():
			if len(ev) > 0:
				if ev in ['esc', 'escape']:
					self.events.append(
						[-99, clock.getTime() - self.start_time])
					self.stopped = True
					self.session.stopped = True
					print('run canceled by user')

				elif ev == 'space':
					self.events.append(
						[99, clock.getTime() - self.start_time])
					if self.phase == 0:
						self.phase_forward()

				elif ev == 's':
					self.events.append([1,clock.getTime()-self.start_time])
					if self.phase == 3:
						self.parameters.update({'answer':0})
					    self.parameters['correct'] = int(self.parameters['answer'] == self.parameters['signal_present'])
						if self.session.use_parallel:
							port.Out32(portaddress,self.session.p_choice_left_unsure)
							core.wait(self.session.p_width)
							port.Out32(portaddress,0)
						# ---------
						self.phase_forward()

				elif ev == 'k':
					self.events.append([1,clock.getTime()-self.start_time])
					if self.phase == 3:
						self.parameters.update({'answer':1})
						self.parameters['correct'] = int(self.parameters['answer'] == self.parameters['signal_present'])
						if self.session.use_parallel:
							port.Out32(portaddress,self.session.p_choice_right_unsure)
							core.wait(self.session.p_width)
							port.Out32(portaddress,0)

						self.phase_forward()
					 
			super(DetectTrial, self).key_event( event )
		
	def run(self):
		super(DetectTrial, self).run()
		trigger = None
		restart = 0 
		while not self.stopped:
			self.run_time = clock.getTime() - self.start_time
			if tracker_on:
				if not self.fix_bounds[1,0] < self.tracker.sample()[0] < self.fix_bounds[0,0]:
					self.fixLost = True
					self.parameters['restarted'] = 1	
				elif not self.fix_bounds[1,1] < self.tracker.sample()[1] < self.fix_bounds[0,1]:
					self.fixLost = True
					self.parameters['restarted'] = 1	
				else:
					self.fixLost = False
				
			if self.phase == 0:
				self.prestimulation_time = clock.getTime()
				# For all trials that are not FTIB, skip phase 0
				if self.ID != 0 and self.ID % self.session.block_length != 0:
					if ( self.prestimulation_time  - self.start_time ) > self.phase_durations[0]:
						self.phase_forward()
				
			elif self.phase == 1:  # pre-stim; phase is timed
				self.cue_time = clock.getTime()
				if ( self.cue_time - self.prestimulation_time ) > self.phase_durations[1]:
					self.phase_forward()

			elif self.phase == 2:  # stimulus presentation; phase is timed
				self.stim_time = clock.getTime()
				if self.session.use_parallel and not self.p_stim_played:
					if self.parameters['signal_present'] == 0:
						trigger = self.session.p_stimulus_noise
					elif self.parameters['signal_present'] == 1:
						if self.grating.ori == -45:
							trigger = self.session.p_stimulus_left
						elif self.grating.ori == 45:
							trigger = self.session.p_stimulus_right

					port.Out32(portaddress,trigger)
					core.wait(self.session.p_width)
					port.Out32(portaddress,0)
					self.p_stim_played =True
					
				if ( self.stim_time - self.cue_time ) > self.phase_durations[2]: 
					self.phase_forward()

			elif self.phase == 3:              # Decision interval; phase is timed, but aborted at response
				self.answer_time = clock.getTime()
				if self.parameters['answer'] != -1: #end phase when respond
					self.phase_forward()
				if ( self.answer_time  - self.stim_time) > self.phase_durations[3]: #end phase after some time when no response
					self.phase_forward()

			elif self.phase == 4:
				self.delay_time = clock.getTime()
				if self.fixLost == True:
					self.parameters['fix_lost'] = 1
					core.wait(0.5)
				else:			
					if self.parameters['answer'] == -1 and not self.p_a_feedback_played and not self.p_v_feedback_played:
						self.parameters['slow_feedback'] = 1
						if self.session.use_parallel:
							trigger = self.session.p_feedback_visual
							port.Out32(portaddress,trigger)
							core.wait(self.session.p_width)
							port.Out32(portaddress,0)        
						core.wait(0.3)
						self.p_v_feedback_played = True
					
					if (self.delay_time - self.answer_time) > self.phase_durations[4]:
						self.stopped = True
						self.stop()
						return
			
			# events and draw:
			self.event()
			self.draw()
			self.noise.updateNoise()
						
		# we have stopped:
		self.stop()

class DetectSession(EyelinkSession):
	def __init__(self, subject_initials, nr_trials, block_length, tracker_on=False, index_number=1, use_parallel=False):
		super(DetectSession, self).__init__( subject_initials, index_number)
		self.create_screen(  size=[1920, 1080],full_screen = fullscr, background_color = (0.5, 0.5, 0.5), physical_screen_distance = 50, engine = 'pygaze') #,  ,
		self.block_length = block_length
		self.nr_trials = nr_trials
		self.index_number = index_number
		self.create_output_filename()
		
		if tracker_on:
			self.create_tracker(sensitivity_class = 2,sample_rate=500)
			super(DetectSession, self).tracker_setup()

		self.use_parallel = use_parallel
		self.p_width = 5/float(1000)

		# Trigger values		
		self.p_run_start = 126				# Trigger to signal session start 
		self.p_run_end = 126                # Trigger to signal session end 
		self.p_stimulus_left = 8            # Trigger for left-oriented Gabor (+ noise)
		self.p_stimulus_right = 9           # Trigger for right-oriented Gabor (+ noise)
		self.p_stimulus_noise = 10          # Trigger for noise-stimulus
		self.p_choice_left = 16             # Trigger for left button press
		self.p_choice_right = 17            # Trigger for right button press            
		self.p_feedback_visual = 33			# Trigger for visual feedback (when response is too late)

		self.create_yes_no_trials()

	def create_yes_no_trials(self):
		"""creates trial parameters for detection task"""
		# standard parameters (same for all trials):
		self.standard_parameters = {'noise_contrast':0.9,'signal_contrast': 0.1, 'spatial_freq': 0.05,'orientation':-1,'opacity':0.9 }
		
		# create trials in nested for-loop:
		self.trial_parameters_and_durs = []    
		trial_counter = 0
		self.total_duration = 0
		self.signal_present = np.array([0,1]) # 0: target absent, 1: target present

		for i in range(self.nr_trials/self.signal_present.shape[0]):
			for j in range(self.signal_present.shape[0]):
				# phase durations, and iti's:
				phase_durs = [-0.01, 0.15, 0.5, 1.0, np.random.uniform(0.3,0.5)]
				params = self.standard_parameters
				params.update({'signal_present': self.signal_present[j]})

				self.trial_parameters_and_durs.append([params.copy(), np.array(phase_durs)])
				self.total_duration += np.array(phase_durs).sum()
				trial_counter += 1

		# Shuffle the order of trials
		self.run_order = np.argsort(np.random.rand(len(self.trial_parameters_and_durs)))

		# print params:
		print("number trials: %i." % trial_counter)
		print("total duration: %.2f min." % (self.total_duration / 60.0))

	def run(self):
		"""run the session. In this specific experiment, participants should fixate as best as possible. 
		If they lose fixation during specific trials phases, a new trial will be appended to the end of the experiment."""

		self.corrects = []
		self.clock = clock

		# Send session start messages to parallel port and tracker
		if self.use_parallel: 
			port.Out32(portaddress,self.p_run_start)
			core.wait(self.p_width)
			port.Out32(portaddress,0)         
		if tracker_on: 
			self.tracker.status_msg('run started at ' + str(clock.getTime()) + ' trigger ' + str(self.p_run_start) )

		# cycle through trials
		trial_counter = 0		
		while trial_counter < self.nr_trials:
			print(trial)

			# If last trial, add some extra time in the last phase, to make sure that relevant data after the last event is still captured 
			if trial_counter == range(self.nr_trials)[-1]: 
				this_trial = DetectTrial( parameters=self.trial_parameters_and_durs[self.run_order[i]][0], phase_durations=[-0.01, 0.3, 0.5, 0.5, 1.0, 5 ], session=self, screen=self.screen, tracker=self.tracker, ID=trial_counter)
			else:      
				this_trial = DetectTrial(parameters=self.trial_parameters_and_durs[self.run_order[i]][0], phase_durations=self.trial_parameters_and_durs[self.run_order[i]][1], session=self, screen=self.screen, tracker=self.tracker, ID=trial_counter)
			
			# run the current trial	
			this_trial.run()	

			# If fixation was lost during this trial, create a new trial with random parameters and append
			if this_trial.parameters['restarted']:
				self.trial_parameters_and_durs.append(self.trial_parameters_and_durs[0])
				self.trial_parameters_and_durs[-1][0]['signal_present'] = np.random.randint(0,2)
				self.nr_trials = self.nr_trials + 1
				self.run_order = np.append( self.run_order, self.run_order.shape[0])
			self.corrects.append(this_trial.parameters['correct'])

			if self.stopped == True:
				break
			trial_counter += 1
		
		# trigger:
		# --------
		if self.use_parallel:
			port.Out32(portaddress,self.p_run_end)
			core.wait(self.p_width)
			port.Out32(portaddress,0)
		# ---------
		if self.tracker_on:
			self.tracker.status_msg('run ended at ' + str(clock.getTime()) + ' trigger ' + str(self.p_run_end) )
		
		self.screen.clearBuffer
		self.close()

def main(initials, index_number, NR_TRIALS):
    #appnope.nope()  # Shut down MAC-OS application app-nap, which shuts down programs after idling for to long
    ts = DetectSession(subject_initials=initials, nr_trials=NR_TRIALS, block_length = block_length,  index_number=index_number, tracker_on=tracker_on, use_parallel=use_parallel)
    ts.run()

    if not os.path.exists('data/detect/' + initials + '/'):
        os.makedirs('data/detect/' + initials + '/')
    shutil.move(os.getcwd() +  '\\' + ts.output_file + '_outputDict.pkl', os.getcwd() +   '\\data\\detect\\' + initials + '\\' + os.path.split(ts.output_file)[1] + '_outputDict.pickle')
    
if __name__ == '__main__':
    # Store info about the experiment session
    initials = raw_input('Participant: ') 
    index_number = int(raw_input('Which run: ')) 
	
    # print(initials)
    main(initials=initials, index_number=index_number, NR_TRIALS=NR_TRIALS)

