#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.5),
    on Fr 21 Mär 2025 10:31:29 CET
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '0'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from functions
import os.path as osp
def get_image_name(filename):
    return osp.splitext(osp.basename(filename))[0]
    
# define a log function that flushes directly
def log(*msgs, sep=' ', end='\n'):
    print(*msgs, sep=sep, end=end, flush=True)
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.5'
expName = 'EMS-TMR-localizer'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '0',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1080, 1920]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/home/simon/zi_nextcloud/Masterthesis_EMS-TMR/EMS-TMR-Experiment/EMS-TMR-localizer.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('data')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=1,
            winType='pyglet', allowGUI=False, allowStencil=True,
            monitor='testMonitor', color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units=None,
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1.0000, 1.0000, 1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = None
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('choice_key') is None:
        # initialise choice_key
        choice_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='choice_key',
        )
    if deviceManager.getDevice('key_resp_8') is None:
        # initialise key_resp_8
        key_resp_8 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_8',
        )
    if deviceManager.getDevice('key_resp_9') is None:
        # initialise key_resp_9
        key_resp_9 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_9',
        )
    if deviceManager.getDevice('key_resp_3') is None:
        # initialise key_resp_3
        key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_3',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('key_resp_localizer_2') is None:
        # initialise key_resp_localizer_2
        key_resp_localizer_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_localizer_2',
        )
    # create speaker 'sound_wrong_2'
    deviceManager.addDevice(
        deviceName='sound_wrong_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'sound_correct_2'
    deviceManager.addDevice(
        deviceName='sound_correct_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_resp_4') is None:
        # initialise key_resp_4
        key_resp_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_4',
        )
    if deviceManager.getDevice('key_resp_localizer') is None:
        # initialise key_resp_localizer
        key_resp_localizer = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_localizer',
        )
    # create speaker 'sound_wrong'
    deviceManager.addDevice(
        deviceName='sound_wrong',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'sound_correct'
    deviceManager.addDevice(
        deviceName='sound_correct',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_resp_10') is None:
        # initialise key_resp_10
        key_resp_10 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_10',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "startup_code" ---
    # Run 'Begin Experiment' code from settings
    t_img_sequence = 0.1
    t_img_localizer = 0.5
    t_isi_localizer = 2.5
    image_size = 0.5
    t_fixation_cross = 1.5
    
     # show break after these trials
    breaks_after_block = [0,1,2]
    
    n_localizer_trials_prac = 6
    n_localizer_trials = 64
    cross_color = (-1.0000, -1.0000, -1.0000)
    letter_height=0.06
    language = "german"
    
    trigger_img = {}
    trigger_img['dog'] = 1
    trigger_img['car']   = 2
    trigger_img['flower']   = 3
    trigger_img['face']   = 4
    
    trigger_break_start = 91
    trigger_break_stop = 92
    trigger_fixation = 81
    
    # these get added depending on the trial
    # ie. Gesicht in localizer = 1, as cue = 11, as sequence=21
    trigger_base_val_localizer = 0
    trigger_base_val_localizer_distractor = 100
    
    # Run 'Begin Experiment' code from startup
    import pandas as pd
    
    subj_id  = f"{int(expInfo['participant']):02d}"
    
    df_localizer = pd.read_excel(f"sequences/{subj_id}_localizer.xlsx")
    
    # set variables we will access later
    i_localizer = 0
    i_block = 0
    
    # store answers
    false_alarms = 0
    misses = 0
    wrong_answers = 0
    
    # set the number of repetitions we have
    n_localizer_trials_prac = 6
    n_blocks = max(df_localizer.block)+1
    n_localizer_trials = len(df_localizer)
    
    if PILOTING:
        n_localizer_trials = 5
        n_block = 2
        print('DUMMY TRIALS: subj_id is set to 0')
    
    
    
    
    # --- Initialize components for Routine "language_selection_screen" ---
    english_polygon = visual.ShapeStim(
        win=win, name='english_polygon',
        size=(0.3, 0.3), vertices='circle',
        ori=0.0, pos=(0.5, 0), draggable=False, anchor='center',
        lineWidth=1.5,
        colorSpace='rgb', lineColor='white', fillColor=None,
        opacity=None, depth=0.0, interpolate=True)
    german_polygon = visual.ShapeStim(
        win=win, name='german_polygon',
        size=(0.3, 0.3), vertices='circle',
        ori=0.0, pos=(-0.5, 0), draggable=False, anchor='center',
        lineWidth=1.5,
        colorSpace='rgb', lineColor='white', fillColor=None,
        opacity=None, depth=-1.0, interpolate=True)
    german_flag = visual.ImageStim(
        win=win,
        name='german_flag', 
        image='stimuli/germanflag.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.5, 0), draggable=False, size=(0.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=True, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    english_flag = visual.ImageStim(
        win=win,
        name='english_flag', 
        image='stimuli/englishflag.png', mask=None, anchor='center',
        ori=0.0, pos=(0.5, 0), draggable=False, size=(0.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    german_text = visual.TextStim(win=win, name='german_text',
        text='Drücke bitte die linke (grüne) Taste für Deutsch.',
        font='Times New Roman',
        pos=(-0.5, -0.25), draggable=False, height=letter_height, wrapWidth=None, ori=0.0, 
        color=[-1.0000, 1.0000, -0.0039], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    english_text = visual.TextStim(win=win, name='english_text',
        text='Press the right (blue) button for English.',
        font='Times New Roman',
        pos=(0.5, -0.25), draggable=False, height=letter_height, wrapWidth=None, ori=0.0, 
        color=[-0.2157, 0.1686, 0.8588], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    choice_key = keyboard.Keyboard(deviceName='choice_key')
    
    # --- Initialize components for Routine "WelcomeScreen" ---
    willkommenstext = visual.TextStim(win=win, name='willkommenstext',
        text='Wilkommen zum Experiment:\n"Gedächtniskonsolidierung im Schlaf"\n\nVielen Dank, dass du mich bei meiner Masterarbeit unterstützt.\nBitte drücke einen beliebigen Knopf, um fortzufahren.',
        font='Times New Roman',
        pos=(0, 0.5), draggable=False, height=letter_height, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    img_welcome = visual.ImageStim(
        win=win,
        name='img_welcome', 
        image='stimuli/Welcome.webp', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.2), draggable=False, size=(1,1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    key_resp_8 = keyboard.Keyboard(deviceName='key_resp_8')
    # Run 'Begin Experiment' code from code_5
    if language=="english":
        willkommenstext.text = """Welcome to the experiment:
            "Memory consolidation in sleep"
            Thank you very much for supporting me in my masterthesis.
            Please press any button to continue."""
    
    # --- Initialize components for Routine "instructions_overview" ---
    instruct_overview = visual.TextStim(win=win, name='instruct_overview',
        text='Du wirst nun in der nächsten Stunde mehrere Aufgaben erledigen. Insgesamt wirst du mit drei unterschiedlichen Aufgaben konfrontiert. Danach wirst du dich ausruhen können und einen Mittagschlaf machen können. Im Anschluss an den Mittagschlaf wird dein erarbeitetes Wissen noch einmal geprüft und dann hast du es auch schon geschafft. \n\nDrücke nun einen beliebigen Knopf, um mit einer Einführung in die erste Aufgabe zu starten.',
        font='Times New Roman',
        pos=(0, 0), draggable=False, height=letter_height, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_9 = keyboard.Keyboard(deviceName='key_resp_9')
    # Run 'Begin Experiment' code from code_6
    if language =='english':
        instruct_overview.text = """ You will complete several tasks in the next hour. 
        You will be faced with a total of three different tasks. Afterwards, you will be able to rest and take a nap. After the nap, the knowledge you have acquired will be tested again. 
        Now press any button to start with an introduction to the first task."""
    
    # --- Initialize components for Routine "buff" ---
    image_6 = visual.ImageStim(
        win=win,
        name='image_6', 
        image='stimuli/buff.webp', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "instruct_pre1" ---
    key_resp_3 = keyboard.Keyboard(deviceName='key_resp_3')
    text_2 = visual.TextStim(win=win, name='text_2',
        text='\nWillkommen zum ersten Teil des Experiments.\n\nIn dieser Aufgabe werden dir nacheinander Bilder gezeigt. Immer wenn ein Bild auf den Kopf gedreht ist, ist es deine Aufgabe so schnell wie möglich eine Taste zu drücken. \nVersuche aufmerksam zu bleiben! \nBitte antworte jeweils so schnell und genau wie möglich. Wenn das Bild richtig herum ist, drücke bitte keine Taste. Du hast bei jedem Bild 1,5 Sekunden Zeit zu antworten.\nDrücke eine Taste um fortzufahren.',
        font='Times New Roman',
        pos=(0, 0), draggable=False, height=letter_height, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "buff" ---
    image_6 = visual.ImageStim(
        win=win,
        name='image_6', 
        image='stimuli/buff.webp', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "instruct_pre2" ---
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    text_4 = visual.TextStim(win=win, name='text_4',
        text='Für eine gute Datenqualität ist es wichtig, dass du während des Experimentes so ruhig wie möglich liegen bleibst. Versuche insbesondere Kopfbewegungen zu minimieren und wenn möglich mit Bewegungen bis zu den Pausen zu warten.\n\nEs wird nun zunächst ein Übungsdurchlauf folgen.\n\nDrücke bitte eine Taste um mit einer kurzen Übung der Aufgabe zu starten.',
        font='Times New Roman',
        pos=(0, 0), draggable=False, height=letter_height, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "fixation_cross" ---
    localizer_fixation = visual.TextStim(win=win, name='localizer_fixation',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.5, wrapWidth=None, ori=0.0, 
        color=cross_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "localizer_prac" ---
    localizer_img_2 = visual.ImageStim(
        win=win,
        name='localizer_img_2', units='height', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=image_size,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    localizer_isi_2 = visual.TextStim(win=win, name='localizer_isi_2',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=letter_height, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_localizer_2 = keyboard.Keyboard(deviceName='key_resp_localizer_2')
    sound_wrong_2 = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_wrong_2',    name='sound_wrong_2'
    )
    sound_wrong_2.setVolume(1.0)
    sound_correct_2 = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_correct_2',    name='sound_correct_2'
    )
    sound_correct_2.setVolume(1.0)
    
    # --- Initialize components for Routine "localizer_feedback" ---
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    text_feedback = visual.TextStim(win=win, name='text_feedback',
        text='... dummy ... something went wrong',
        font='Times New Roman',
        pos=(0, 0), draggable=False, height=letter_height, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "counting_code" ---
    
    # --- Initialize components for Routine "fixation_cross" ---
    localizer_fixation = visual.TextStim(win=win, name='localizer_fixation',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.5, wrapWidth=None, ori=0.0, 
        color=cross_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "localizer" ---
    localizer_img = visual.ImageStim(
        win=win,
        name='localizer_img', units='height', 
        image='stimuli/face/face000.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=image_size,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    localizer_isi = visual.TextStim(win=win, name='localizer_isi',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=letter_height, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_localizer = keyboard.Keyboard(deviceName='key_resp_localizer')
    sound_wrong = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_wrong',    name='sound_wrong'
    )
    sound_wrong.setVolume(1.0)
    sound_correct = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_correct',    name='sound_correct'
    )
    sound_correct.setVolume(1.0)
    
    # --- Initialize components for Routine "break_3" ---
    text = visual.TextStim(win=win, name='text',
        text='Eine kurze Pause.\n\nBitte nimm eine kurze Verschnaufpause. Sobald du dich wieder bereit fühlst, drücke bitte eine beliebige Taste, um mit der Aufgabe fortzufahren.',
        font='Times New Roman',
        pos=(0, 0), draggable=False, height=letter_height, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_10 = keyboard.Keyboard(deviceName='key_resp_10')
    
    # --- Initialize components for Routine "instruct_end" ---
    endtask1 = visual.TextStim(win=win, name='endtask1',
        text='Die Übung ist nun beendet. \n\nBitte melde dich beim Experimentleiter.',
        font='Times New Roman',
        pos=(0, 0), draggable=False, height=letter_height, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from code_7
    if language == "english":
        endtask1 = """ The experiment has ended.
        Please let the experimentor know that you are finished."""
    
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "startup_code" ---
    # create an object to store info about Routine startup_code
    startup_code = data.Routine(
        name='startup_code',
        components=[],
    )
    startup_code.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for startup_code
    startup_code.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    startup_code.tStart = globalClock.getTime(format='float')
    startup_code.status = STARTED
    thisExp.addData('startup_code.started', startup_code.tStart)
    startup_code.maxDuration = None
    # keep track of which components have finished
    startup_codeComponents = startup_code.components
    for thisComponent in startup_code.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "startup_code" ---
    startup_code.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            startup_code.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in startup_code.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "startup_code" ---
    for thisComponent in startup_code.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for startup_code
    startup_code.tStop = globalClock.getTime(format='float')
    startup_code.tStopRefresh = tThisFlipGlobal
    thisExp.addData('startup_code.stopped', startup_code.tStop)
    thisExp.nextEntry()
    # the Routine "startup_code" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "language_selection_screen" ---
    # create an object to store info about Routine language_selection_screen
    language_selection_screen = data.Routine(
        name='language_selection_screen',
        components=[english_polygon, german_polygon, german_flag, english_flag, german_text, english_text, choice_key],
    )
    language_selection_screen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for choice_key
    choice_key.keys = []
    choice_key.rt = []
    _choice_key_allKeys = []
    # Run 'Begin Routine' code from choice_code
    win.mouseVisible = False
    # store start times for language_selection_screen
    language_selection_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    language_selection_screen.tStart = globalClock.getTime(format='float')
    language_selection_screen.status = STARTED
    thisExp.addData('language_selection_screen.started', language_selection_screen.tStart)
    language_selection_screen.maxDuration = None
    # keep track of which components have finished
    language_selection_screenComponents = language_selection_screen.components
    for thisComponent in language_selection_screen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "language_selection_screen" ---
    language_selection_screen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *english_polygon* updates
        
        # if english_polygon is starting this frame...
        if english_polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            english_polygon.frameNStart = frameN  # exact frame index
            english_polygon.tStart = t  # local t and not account for scr refresh
            english_polygon.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(english_polygon, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'english_polygon.started')
            # update status
            english_polygon.status = STARTED
            english_polygon.setAutoDraw(True)
        
        # if english_polygon is active this frame...
        if english_polygon.status == STARTED:
            # update params
            pass
        
        # *german_polygon* updates
        
        # if german_polygon is starting this frame...
        if german_polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            german_polygon.frameNStart = frameN  # exact frame index
            german_polygon.tStart = t  # local t and not account for scr refresh
            german_polygon.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(german_polygon, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'german_polygon.started')
            # update status
            german_polygon.status = STARTED
            german_polygon.setAutoDraw(True)
        
        # if german_polygon is active this frame...
        if german_polygon.status == STARTED:
            # update params
            pass
        
        # *german_flag* updates
        
        # if german_flag is starting this frame...
        if german_flag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            german_flag.frameNStart = frameN  # exact frame index
            german_flag.tStart = t  # local t and not account for scr refresh
            german_flag.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(german_flag, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'german_flag.started')
            # update status
            german_flag.status = STARTED
            german_flag.setAutoDraw(True)
        
        # if german_flag is active this frame...
        if german_flag.status == STARTED:
            # update params
            pass
        
        # *english_flag* updates
        
        # if english_flag is starting this frame...
        if english_flag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            english_flag.frameNStart = frameN  # exact frame index
            english_flag.tStart = t  # local t and not account for scr refresh
            english_flag.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(english_flag, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'english_flag.started')
            # update status
            english_flag.status = STARTED
            english_flag.setAutoDraw(True)
        
        # if english_flag is active this frame...
        if english_flag.status == STARTED:
            # update params
            pass
        
        # *german_text* updates
        
        # if german_text is starting this frame...
        if german_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            german_text.frameNStart = frameN  # exact frame index
            german_text.tStart = t  # local t and not account for scr refresh
            german_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(german_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'german_text.started')
            # update status
            german_text.status = STARTED
            german_text.setAutoDraw(True)
        
        # if german_text is active this frame...
        if german_text.status == STARTED:
            # update params
            pass
        
        # *english_text* updates
        
        # if english_text is starting this frame...
        if english_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            english_text.frameNStart = frameN  # exact frame index
            english_text.tStart = t  # local t and not account for scr refresh
            english_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(english_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'english_text.started')
            # update status
            english_text.status = STARTED
            english_text.setAutoDraw(True)
        
        # if english_text is active this frame...
        if english_text.status == STARTED:
            # update params
            pass
        
        # *choice_key* updates
        waitOnFlip = False
        
        # if choice_key is starting this frame...
        if choice_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            choice_key.frameNStart = frameN  # exact frame index
            choice_key.tStart = t  # local t and not account for scr refresh
            choice_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(choice_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'choice_key.started')
            # update status
            choice_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(choice_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(choice_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if choice_key.status == STARTED and not waitOnFlip:
            theseKeys = choice_key.getKeys(keyList=['g','b', 'r'], ignoreKeys=["escape"], waitRelease=True)
            _choice_key_allKeys.extend(theseKeys)
            if len(_choice_key_allKeys):
                choice_key.keys = _choice_key_allKeys[-1].name  # just the last key pressed
                choice_key.rt = _choice_key_allKeys[-1].rt
                choice_key.duration = _choice_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            language_selection_screen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in language_selection_screen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "language_selection_screen" ---
    for thisComponent in language_selection_screen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for language_selection_screen
    language_selection_screen.tStop = globalClock.getTime(format='float')
    language_selection_screen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('language_selection_screen.stopped', language_selection_screen.tStop)
    # check responses
    if choice_key.keys in ['', [], None]:  # No response was made
        choice_key.keys = None
    thisExp.addData('choice_key.keys',choice_key.keys)
    if choice_key.keys != None:  # we had a response
        thisExp.addData('choice_key.rt', choice_key.rt)
        thisExp.addData('choice_key.duration', choice_key.duration)
    # Run 'End Routine' code from choice_code
    
    if choice_key.keys == "g":
        german_polygon.borderColor = [1, 0.2941, -1]
        language = "german"
    elif choice_key.keys in "br":
        english_polygon.borderColor = [1, 0.2941, -1]
        language = "english"
    
    thisExp.nextEntry()
    # the Routine "language_selection_screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "WelcomeScreen" ---
    # create an object to store info about Routine WelcomeScreen
    WelcomeScreen = data.Routine(
        name='WelcomeScreen',
        components=[willkommenstext, img_welcome, key_resp_8],
    )
    WelcomeScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_8
    key_resp_8.keys = []
    key_resp_8.rt = []
    _key_resp_8_allKeys = []
    # store start times for WelcomeScreen
    WelcomeScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    WelcomeScreen.tStart = globalClock.getTime(format='float')
    WelcomeScreen.status = STARTED
    thisExp.addData('WelcomeScreen.started', WelcomeScreen.tStart)
    WelcomeScreen.maxDuration = None
    # keep track of which components have finished
    WelcomeScreenComponents = WelcomeScreen.components
    for thisComponent in WelcomeScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "WelcomeScreen" ---
    WelcomeScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 60.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *willkommenstext* updates
        
        # if willkommenstext is starting this frame...
        if willkommenstext.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            willkommenstext.frameNStart = frameN  # exact frame index
            willkommenstext.tStart = t  # local t and not account for scr refresh
            willkommenstext.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(willkommenstext, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'willkommenstext.started')
            # update status
            willkommenstext.status = STARTED
            willkommenstext.setAutoDraw(True)
        
        # if willkommenstext is active this frame...
        if willkommenstext.status == STARTED:
            # update params
            pass
        
        # if willkommenstext is stopping this frame...
        if willkommenstext.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > willkommenstext.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                willkommenstext.tStop = t  # not accounting for scr refresh
                willkommenstext.tStopRefresh = tThisFlipGlobal  # on global time
                willkommenstext.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'willkommenstext.stopped')
                # update status
                willkommenstext.status = FINISHED
                willkommenstext.setAutoDraw(False)
        
        # *img_welcome* updates
        
        # if img_welcome is starting this frame...
        if img_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img_welcome.frameNStart = frameN  # exact frame index
            img_welcome.tStart = t  # local t and not account for scr refresh
            img_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img_welcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img_welcome.started')
            # update status
            img_welcome.status = STARTED
            img_welcome.setAutoDraw(True)
        
        # if img_welcome is active this frame...
        if img_welcome.status == STARTED:
            # update params
            pass
        
        # if img_welcome is stopping this frame...
        if img_welcome.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > img_welcome.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                img_welcome.tStop = t  # not accounting for scr refresh
                img_welcome.tStopRefresh = tThisFlipGlobal  # on global time
                img_welcome.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img_welcome.stopped')
                # update status
                img_welcome.status = FINISHED
                img_welcome.setAutoDraw(False)
        
        # *key_resp_8* updates
        waitOnFlip = False
        
        # if key_resp_8 is starting this frame...
        if key_resp_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_8.frameNStart = frameN  # exact frame index
            key_resp_8.tStart = t  # local t and not account for scr refresh
            key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_8.started')
            # update status
            key_resp_8.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_8.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp_8 is stopping this frame...
        if key_resp_8.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp_8.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                key_resp_8.tStop = t  # not accounting for scr refresh
                key_resp_8.tStopRefresh = tThisFlipGlobal  # on global time
                key_resp_8.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_8.stopped')
                # update status
                key_resp_8.status = FINISHED
                key_resp_8.status = FINISHED
        if key_resp_8.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_8.getKeys(keyList=['y','b','r', 'g'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_8_allKeys.extend(theseKeys)
            if len(_key_resp_8_allKeys):
                key_resp_8.keys = _key_resp_8_allKeys[-1].name  # just the last key pressed
                key_resp_8.rt = _key_resp_8_allKeys[-1].rt
                key_resp_8.duration = _key_resp_8_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            WelcomeScreen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in WelcomeScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "WelcomeScreen" ---
    for thisComponent in WelcomeScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for WelcomeScreen
    WelcomeScreen.tStop = globalClock.getTime(format='float')
    WelcomeScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('WelcomeScreen.stopped', WelcomeScreen.tStop)
    # check responses
    if key_resp_8.keys in ['', [], None]:  # No response was made
        key_resp_8.keys = None
    thisExp.addData('key_resp_8.keys',key_resp_8.keys)
    if key_resp_8.keys != None:  # we had a response
        thisExp.addData('key_resp_8.rt', key_resp_8.rt)
        thisExp.addData('key_resp_8.duration', key_resp_8.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if WelcomeScreen.maxDurationReached:
        routineTimer.addTime(-WelcomeScreen.maxDuration)
    elif WelcomeScreen.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-60.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "instructions_overview" ---
    # create an object to store info about Routine instructions_overview
    instructions_overview = data.Routine(
        name='instructions_overview',
        components=[instruct_overview, key_resp_9],
    )
    instructions_overview.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_9
    key_resp_9.keys = []
    key_resp_9.rt = []
    _key_resp_9_allKeys = []
    # store start times for instructions_overview
    instructions_overview.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_overview.tStart = globalClock.getTime(format='float')
    instructions_overview.status = STARTED
    thisExp.addData('instructions_overview.started', instructions_overview.tStart)
    instructions_overview.maxDuration = None
    # keep track of which components have finished
    instructions_overviewComponents = instructions_overview.components
    for thisComponent in instructions_overview.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_overview" ---
    instructions_overview.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 60.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruct_overview* updates
        
        # if instruct_overview is starting this frame...
        if instruct_overview.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruct_overview.frameNStart = frameN  # exact frame index
            instruct_overview.tStart = t  # local t and not account for scr refresh
            instruct_overview.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruct_overview, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruct_overview.started')
            # update status
            instruct_overview.status = STARTED
            instruct_overview.setAutoDraw(True)
        
        # if instruct_overview is active this frame...
        if instruct_overview.status == STARTED:
            # update params
            pass
        
        # if instruct_overview is stopping this frame...
        if instruct_overview.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > instruct_overview.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                instruct_overview.tStop = t  # not accounting for scr refresh
                instruct_overview.tStopRefresh = tThisFlipGlobal  # on global time
                instruct_overview.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instruct_overview.stopped')
                # update status
                instruct_overview.status = FINISHED
                instruct_overview.setAutoDraw(False)
        
        # *key_resp_9* updates
        waitOnFlip = False
        
        # if key_resp_9 is starting this frame...
        if key_resp_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_9.frameNStart = frameN  # exact frame index
            key_resp_9.tStart = t  # local t and not account for scr refresh
            key_resp_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_9.started')
            # update status
            key_resp_9.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_9.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp_9 is stopping this frame...
        if key_resp_9.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp_9.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                key_resp_9.tStop = t  # not accounting for scr refresh
                key_resp_9.tStopRefresh = tThisFlipGlobal  # on global time
                key_resp_9.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_9.stopped')
                # update status
                key_resp_9.status = FINISHED
                key_resp_9.status = FINISHED
        if key_resp_9.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_9.getKeys(keyList=['y','b','r', 'g'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_9_allKeys.extend(theseKeys)
            if len(_key_resp_9_allKeys):
                key_resp_9.keys = _key_resp_9_allKeys[-1].name  # just the last key pressed
                key_resp_9.rt = _key_resp_9_allKeys[-1].rt
                key_resp_9.duration = _key_resp_9_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions_overview.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_overview.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_overview" ---
    for thisComponent in instructions_overview.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_overview
    instructions_overview.tStop = globalClock.getTime(format='float')
    instructions_overview.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_overview.stopped', instructions_overview.tStop)
    # check responses
    if key_resp_9.keys in ['', [], None]:  # No response was made
        key_resp_9.keys = None
    thisExp.addData('key_resp_9.keys',key_resp_9.keys)
    if key_resp_9.keys != None:  # we had a response
        thisExp.addData('key_resp_9.rt', key_resp_9.rt)
        thisExp.addData('key_resp_9.duration', key_resp_9.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if instructions_overview.maxDurationReached:
        routineTimer.addTime(-instructions_overview.maxDuration)
    elif instructions_overview.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-60.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "buff" ---
    # create an object to store info about Routine buff
    buff = data.Routine(
        name='buff',
        components=[image_6],
    )
    buff.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for buff
    buff.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    buff.tStart = globalClock.getTime(format='float')
    buff.status = STARTED
    thisExp.addData('buff.started', buff.tStart)
    buff.maxDuration = None
    # keep track of which components have finished
    buffComponents = buff.components
    for thisComponent in buff.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "buff" ---
    buff.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.8:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_6* updates
        
        # if image_6 is starting this frame...
        if image_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_6.frameNStart = frameN  # exact frame index
            image_6.tStart = t  # local t and not account for scr refresh
            image_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_6.started')
            # update status
            image_6.status = STARTED
            image_6.setAutoDraw(True)
        
        # if image_6 is active this frame...
        if image_6.status == STARTED:
            # update params
            pass
        
        # if image_6 is stopping this frame...
        if image_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_6.tStartRefresh + 0.8-frameTolerance:
                # keep track of stop time/frame for later
                image_6.tStop = t  # not accounting for scr refresh
                image_6.tStopRefresh = tThisFlipGlobal  # on global time
                image_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_6.stopped')
                # update status
                image_6.status = FINISHED
                image_6.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            buff.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in buff.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "buff" ---
    for thisComponent in buff.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for buff
    buff.tStop = globalClock.getTime(format='float')
    buff.tStopRefresh = tThisFlipGlobal
    thisExp.addData('buff.stopped', buff.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if buff.maxDurationReached:
        routineTimer.addTime(-buff.maxDuration)
    elif buff.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.800000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "instruct_pre1" ---
    # create an object to store info about Routine instruct_pre1
    instruct_pre1 = data.Routine(
        name='instruct_pre1',
        components=[key_resp_3, text_2],
    )
    instruct_pre1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_3
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # store start times for instruct_pre1
    instruct_pre1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_pre1.tStart = globalClock.getTime(format='float')
    instruct_pre1.status = STARTED
    thisExp.addData('instruct_pre1.started', instruct_pre1.tStart)
    instruct_pre1.maxDuration = None
    # keep track of which components have finished
    instruct_pre1Components = instruct_pre1.components
    for thisComponent in instruct_pre1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_pre1" ---
    instruct_pre1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 60.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp_3 is stopping this frame...
        if key_resp_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp_3.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                key_resp_3.tStop = t  # not accounting for scr refresh
                key_resp_3.tStopRefresh = tThisFlipGlobal  # on global time
                key_resp_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_3.stopped')
                # update status
                key_resp_3.status = FINISHED
                key_resp_3.status = FINISHED
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['y','b','r', 'g'], ignoreKeys=["escape"], waitRelease=True)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # if text_2 is stopping this frame...
        if text_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_2.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                text_2.tStop = t  # not accounting for scr refresh
                text_2.tStopRefresh = tThisFlipGlobal  # on global time
                text_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.stopped')
                # update status
                text_2.status = FINISHED
                text_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instruct_pre1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_pre1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_pre1" ---
    for thisComponent in instruct_pre1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_pre1
    instruct_pre1.tStop = globalClock.getTime(format='float')
    instruct_pre1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruct_pre1.stopped', instruct_pre1.tStop)
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if instruct_pre1.maxDurationReached:
        routineTimer.addTime(-instruct_pre1.maxDuration)
    elif instruct_pre1.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-60.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "buff" ---
    # create an object to store info about Routine buff
    buff = data.Routine(
        name='buff',
        components=[image_6],
    )
    buff.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for buff
    buff.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    buff.tStart = globalClock.getTime(format='float')
    buff.status = STARTED
    thisExp.addData('buff.started', buff.tStart)
    buff.maxDuration = None
    # keep track of which components have finished
    buffComponents = buff.components
    for thisComponent in buff.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "buff" ---
    buff.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.8:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_6* updates
        
        # if image_6 is starting this frame...
        if image_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_6.frameNStart = frameN  # exact frame index
            image_6.tStart = t  # local t and not account for scr refresh
            image_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_6.started')
            # update status
            image_6.status = STARTED
            image_6.setAutoDraw(True)
        
        # if image_6 is active this frame...
        if image_6.status == STARTED:
            # update params
            pass
        
        # if image_6 is stopping this frame...
        if image_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_6.tStartRefresh + 0.8-frameTolerance:
                # keep track of stop time/frame for later
                image_6.tStop = t  # not accounting for scr refresh
                image_6.tStopRefresh = tThisFlipGlobal  # on global time
                image_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_6.stopped')
                # update status
                image_6.status = FINISHED
                image_6.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            buff.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in buff.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "buff" ---
    for thisComponent in buff.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for buff
    buff.tStop = globalClock.getTime(format='float')
    buff.tStopRefresh = tThisFlipGlobal
    thisExp.addData('buff.stopped', buff.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if buff.maxDurationReached:
        routineTimer.addTime(-buff.maxDuration)
    elif buff.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.800000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "instruct_pre2" ---
    # create an object to store info about Routine instruct_pre2
    instruct_pre2 = data.Routine(
        name='instruct_pre2',
        components=[key_resp_2, text_4],
    )
    instruct_pre2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # Run 'Begin Routine' code from code_4
    if language=='english':
        text_4.text=f"""To ensure good data quality, it is important that you remain as still as possible during the experiment. In particular, try to minimize head movements and, if possible, wait until the breaks before making any movements.
        Please try to answer as quickly as possible.A short practice trial will follow now.
        Press any button to start the practice trial."""
    
    # store start times for instruct_pre2
    instruct_pre2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_pre2.tStart = globalClock.getTime(format='float')
    instruct_pre2.status = STARTED
    thisExp.addData('instruct_pre2.started', instruct_pre2.tStart)
    instruct_pre2.maxDuration = None
    # keep track of which components have finished
    instruct_pre2Components = instruct_pre2.components
    for thisComponent in instruct_pre2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_pre2" ---
    instruct_pre2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 60.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp_2 is stopping this frame...
        if key_resp_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp_2.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                key_resp_2.tStop = t  # not accounting for scr refresh
                key_resp_2.tStopRefresh = tThisFlipGlobal  # on global time
                key_resp_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_2.stopped')
                # update status
                key_resp_2.status = FINISHED
                key_resp_2.status = FINISHED
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['y','b','r', 'g'], ignoreKeys=["escape"], waitRelease=True)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_4.started')
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # if text_4 is stopping this frame...
        if text_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_4.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                text_4.tStop = t  # not accounting for scr refresh
                text_4.tStopRefresh = tThisFlipGlobal  # on global time
                text_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_4.stopped')
                # update status
                text_4.status = FINISHED
                text_4.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instruct_pre2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_pre2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_pre2" ---
    for thisComponent in instruct_pre2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_pre2
    instruct_pre2.tStop = globalClock.getTime(format='float')
    instruct_pre2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruct_pre2.stopped', instruct_pre2.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if instruct_pre2.maxDurationReached:
        routineTimer.addTime(-instruct_pre2.maxDuration)
    elif instruct_pre2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-60.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    localizer_repetition = data.TrialHandler2(
        name='localizer_repetition',
        nReps=0.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(localizer_repetition)  # add the loop to the experiment
    thisLocalizer_repetition = localizer_repetition.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLocalizer_repetition.rgb)
    if thisLocalizer_repetition != None:
        for paramName in thisLocalizer_repetition:
            globals()[paramName] = thisLocalizer_repetition[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisLocalizer_repetition in localizer_repetition:
        currentLoop = localizer_repetition
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisLocalizer_repetition.rgb)
        if thisLocalizer_repetition != None:
            for paramName in thisLocalizer_repetition:
                globals()[paramName] = thisLocalizer_repetition[paramName]
        
        # set up handler to look after randomisation of conditions etc
        localizer_trials_prac = data.TrialHandler2(
            name='localizer_trials_prac',
            nReps=n_localizer_trials_prac, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(localizer_trials_prac)  # add the loop to the experiment
        thisLocalizer_trials_prac = localizer_trials_prac.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLocalizer_trials_prac.rgb)
        if thisLocalizer_trials_prac != None:
            for paramName in thisLocalizer_trials_prac:
                globals()[paramName] = thisLocalizer_trials_prac[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisLocalizer_trials_prac in localizer_trials_prac:
            currentLoop = localizer_trials_prac
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisLocalizer_trials_prac.rgb)
            if thisLocalizer_trials_prac != None:
                for paramName in thisLocalizer_trials_prac:
                    globals()[paramName] = thisLocalizer_trials_prac[paramName]
            
            # --- Prepare to start Routine "fixation_cross" ---
            # create an object to store info about Routine fixation_cross
            fixation_cross = data.Routine(
                name='fixation_cross',
                components=[localizer_fixation],
            )
            fixation_cross.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for fixation_cross
            fixation_cross.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            fixation_cross.tStart = globalClock.getTime(format='float')
            fixation_cross.status = STARTED
            thisExp.addData('fixation_cross.started', fixation_cross.tStart)
            fixation_cross.maxDuration = None
            # keep track of which components have finished
            fixation_crossComponents = fixation_cross.components
            for thisComponent in fixation_cross.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation_cross" ---
            # if trial has changed, end Routine now
            if isinstance(localizer_trials_prac, data.TrialHandler2) and thisLocalizer_trials_prac.thisN != localizer_trials_prac.thisTrial.thisN:
                continueRoutine = False
            fixation_cross.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *localizer_fixation* updates
                
                # if localizer_fixation is starting this frame...
                if localizer_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    localizer_fixation.frameNStart = frameN  # exact frame index
                    localizer_fixation.tStart = t  # local t and not account for scr refresh
                    localizer_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(localizer_fixation, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'localizer_fixation.started')
                    # update status
                    localizer_fixation.status = STARTED
                    localizer_fixation.setAutoDraw(True)
                
                # if localizer_fixation is active this frame...
                if localizer_fixation.status == STARTED:
                    # update params
                    pass
                
                # if localizer_fixation is stopping this frame...
                if localizer_fixation.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > localizer_fixation.tStartRefresh + t_fixation_cross-frameTolerance:
                        # keep track of stop time/frame for later
                        localizer_fixation.tStop = t  # not accounting for scr refresh
                        localizer_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                        localizer_fixation.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'localizer_fixation.stopped')
                        # update status
                        localizer_fixation.status = FINISHED
                        localizer_fixation.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    fixation_cross.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixation_cross.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation_cross" ---
            for thisComponent in fixation_cross.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for fixation_cross
            fixation_cross.tStop = globalClock.getTime(format='float')
            fixation_cross.tStopRefresh = tThisFlipGlobal
            thisExp.addData('fixation_cross.stopped', fixation_cross.tStop)
            # the Routine "fixation_cross" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "localizer_prac" ---
            # create an object to store info about Routine localizer_prac
            localizer_prac = data.Routine(
                name='localizer_prac',
                components=[localizer_img_2, localizer_isi_2, key_resp_localizer_2, sound_wrong_2, sound_correct_2],
            )
            localizer_prac.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_localizer_2
            key_resp_localizer_2.keys = []
            key_resp_localizer_2.rt = []
            _key_resp_localizer_2_allKeys = []
            # Run 'Begin Routine' code from localizer_code_2
            df_trial = df_localizer.iloc[i_localizer]
            
            localizer_img_2.image = "stimuli/" + df_trial.image
            t_isi_localizer_2 = df_trial.iti
            is_distractor = df_trial.distractor
            
            if i_localizer == 3 and PILOTING:
                   # overwrite for 
                   is_distractor = True
                   
            if is_distractor:
                localizer_img_2.ori  = 180 #image flipped
            else:
                localizer_img_2.ori  = 0
                
            # prevent inactive sound from blocking trial finish
            sound_correct_2.status = FINISHED
            sound_wrong_2.status = FINISHED
            played = False
            
            # debug printing
            msg = f'localizer practice {i_localizer}/{len(df_trial)}'
            msg += f' {"[FLIPPED]" if is_distractor else ""} isi={t_isi_localizer:.2f}s'
            log(msg)
            
            sound_wrong_2.setSound('stimuli/soundError.wav', secs=0.5, hamming=True)
            sound_wrong_2.setVolume(1.0, log=False)
            sound_wrong_2.seek(0)
            sound_correct_2.setSound('stimuli/soundCoin.wav', secs=0.5, hamming=True)
            sound_correct_2.setVolume(1.0, log=False)
            sound_correct_2.seek(0)
            # store start times for localizer_prac
            localizer_prac.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            localizer_prac.tStart = globalClock.getTime(format='float')
            localizer_prac.status = STARTED
            thisExp.addData('localizer_prac.started', localizer_prac.tStart)
            localizer_prac.maxDuration = None
            # keep track of which components have finished
            localizer_pracComponents = localizer_prac.components
            for thisComponent in localizer_prac.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "localizer_prac" ---
            # if trial has changed, end Routine now
            if isinstance(localizer_trials_prac, data.TrialHandler2) and thisLocalizer_trials_prac.thisN != localizer_trials_prac.thisTrial.thisN:
                continueRoutine = False
            localizer_prac.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *localizer_img_2* updates
                
                # if localizer_img_2 is starting this frame...
                if localizer_img_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    localizer_img_2.frameNStart = frameN  # exact frame index
                    localizer_img_2.tStart = t  # local t and not account for scr refresh
                    localizer_img_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(localizer_img_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'localizer_img_2.started')
                    # update status
                    localizer_img_2.status = STARTED
                    localizer_img_2.setAutoDraw(True)
                
                # if localizer_img_2 is active this frame...
                if localizer_img_2.status == STARTED:
                    # update params
                    pass
                
                # if localizer_img_2 is stopping this frame...
                if localizer_img_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > localizer_img_2.tStartRefresh + t_img_localizer-frameTolerance:
                        # keep track of stop time/frame for later
                        localizer_img_2.tStop = t  # not accounting for scr refresh
                        localizer_img_2.tStopRefresh = tThisFlipGlobal  # on global time
                        localizer_img_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'localizer_img_2.stopped')
                        # update status
                        localizer_img_2.status = FINISHED
                        localizer_img_2.setAutoDraw(False)
                
                # *localizer_isi_2* updates
                
                # if localizer_isi_2 is starting this frame...
                if localizer_isi_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    localizer_isi_2.frameNStart = frameN  # exact frame index
                    localizer_isi_2.tStart = t  # local t and not account for scr refresh
                    localizer_isi_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(localizer_isi_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'localizer_isi_2.started')
                    # update status
                    localizer_isi_2.status = STARTED
                    localizer_isi_2.setAutoDraw(True)
                
                # if localizer_isi_2 is active this frame...
                if localizer_isi_2.status == STARTED:
                    # update params
                    pass
                
                # if localizer_isi_2 is stopping this frame...
                if localizer_isi_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > localizer_isi_2.tStartRefresh + t_isi_localizer_2-frameTolerance:
                        # keep track of stop time/frame for later
                        localizer_isi_2.tStop = t  # not accounting for scr refresh
                        localizer_isi_2.tStopRefresh = tThisFlipGlobal  # on global time
                        localizer_isi_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'localizer_isi_2.stopped')
                        # update status
                        localizer_isi_2.status = FINISHED
                        localizer_isi_2.setAutoDraw(False)
                
                # *key_resp_localizer_2* updates
                waitOnFlip = False
                
                # if key_resp_localizer_2 is starting this frame...
                if key_resp_localizer_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_localizer_2.frameNStart = frameN  # exact frame index
                    key_resp_localizer_2.tStart = t  # local t and not account for scr refresh
                    key_resp_localizer_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_localizer_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_localizer_2.started')
                    # update status
                    key_resp_localizer_2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_localizer_2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_localizer_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp_localizer_2 is stopping this frame...
                if key_resp_localizer_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_localizer_2.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_localizer_2.tStop = t  # not accounting for scr refresh
                        key_resp_localizer_2.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_localizer_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_localizer_2.stopped')
                        # update status
                        key_resp_localizer_2.status = FINISHED
                        key_resp_localizer_2.status = FINISHED
                if key_resp_localizer_2.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_localizer_2.getKeys(keyList=['y', 'g', 'b', 'r'], ignoreKeys=["escape"], waitRelease=True)
                    _key_resp_localizer_2_allKeys.extend(theseKeys)
                    if len(_key_resp_localizer_2_allKeys):
                        key_resp_localizer_2.keys = _key_resp_localizer_2_allKeys[-1].name  # just the last key pressed
                        key_resp_localizer_2.rt = _key_resp_localizer_2_allKeys[-1].rt
                        key_resp_localizer_2.duration = _key_resp_localizer_2_allKeys[-1].duration
                # Run 'Each Frame' code from localizer_code_2
                # play for wrong/correct button presses
                if len(key_resp_localizer_2.keys) and not played:
                    played = True
                    if is_distractor:
                        sound_correct_2.play()
                        log('Correct press')
                    else:
                        sound_wrong_2.play()
                        false_alarms += 1
                        log('False alarm')
                
                # play error in case of missed button press
                if is_distractor and key_resp_localizer_2.status==FINISHED and not played:
                    played = True
                    sound_wrong_2.play()
                    misses += 1
                    log('Miss')
                    
                
                sound_correct_2.status = FINISHED
                sound_wrong_2.status = FINISHED
                
                # *sound_wrong_2* updates
                
                # if sound_wrong_2 is starting this frame...
                if sound_wrong_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    sound_wrong_2.frameNStart = frameN  # exact frame index
                    sound_wrong_2.tStart = t  # local t and not account for scr refresh
                    sound_wrong_2.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('sound_wrong_2.started', tThisFlipGlobal)
                    # update status
                    sound_wrong_2.status = STARTED
                    sound_wrong_2.play(when=win)  # sync with win flip
                
                # if sound_wrong_2 is stopping this frame...
                if sound_wrong_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sound_wrong_2.tStartRefresh + 0.5-frameTolerance or sound_wrong_2.isFinished:
                        # keep track of stop time/frame for later
                        sound_wrong_2.tStop = t  # not accounting for scr refresh
                        sound_wrong_2.tStopRefresh = tThisFlipGlobal  # on global time
                        sound_wrong_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sound_wrong_2.stopped')
                        # update status
                        sound_wrong_2.status = FINISHED
                        sound_wrong_2.stop()
                
                # *sound_correct_2* updates
                
                # if sound_correct_2 is starting this frame...
                if sound_correct_2.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                    # keep track of start time/frame for later
                    sound_correct_2.frameNStart = frameN  # exact frame index
                    sound_correct_2.tStart = t  # local t and not account for scr refresh
                    sound_correct_2.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('sound_correct_2.started', tThisFlipGlobal)
                    # update status
                    sound_correct_2.status = STARTED
                    sound_correct_2.play(when=win)  # sync with win flip
                
                # if sound_correct_2 is stopping this frame...
                if sound_correct_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sound_correct_2.tStartRefresh + 0.5-frameTolerance or sound_correct_2.isFinished:
                        # keep track of stop time/frame for later
                        sound_correct_2.tStop = t  # not accounting for scr refresh
                        sound_correct_2.tStopRefresh = tThisFlipGlobal  # on global time
                        sound_correct_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sound_correct_2.stopped')
                        # update status
                        sound_correct_2.status = FINISHED
                        sound_correct_2.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[sound_wrong_2, sound_correct_2]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    localizer_prac.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in localizer_prac.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "localizer_prac" ---
            for thisComponent in localizer_prac.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for localizer_prac
            localizer_prac.tStop = globalClock.getTime(format='float')
            localizer_prac.tStopRefresh = tThisFlipGlobal
            thisExp.addData('localizer_prac.stopped', localizer_prac.tStop)
            # check responses
            if key_resp_localizer_2.keys in ['', [], None]:  # No response was made
                key_resp_localizer_2.keys = None
            localizer_trials_prac.addData('key_resp_localizer_2.keys',key_resp_localizer_2.keys)
            if key_resp_localizer_2.keys != None:  # we had a response
                localizer_trials_prac.addData('key_resp_localizer_2.rt', key_resp_localizer_2.rt)
                localizer_trials_prac.addData('key_resp_localizer_2.duration', key_resp_localizer_2.duration)
            # Run 'End Routine' code from localizer_code_2
            i_localizer+=1
            sound_wrong_2.pause()  # ensure sound has stopped at end of Routine
            sound_correct_2.pause()  # ensure sound has stopped at end of Routine
            # the Routine "localizer_prac" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed n_localizer_trials_prac repeats of 'localizer_trials_prac'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "localizer_feedback" ---
        # create an object to store info about Routine localizer_feedback
        localizer_feedback = data.Routine(
            name='localizer_feedback',
            components=[key_resp_4, text_feedback],
        )
        localizer_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp_4
        key_resp_4.keys = []
        key_resp_4.rt = []
        _key_resp_4_allKeys = []
        # Run 'Begin Routine' code from feedback_loop
        if false_alarms==0 and misses==0:
            localizer_repetition.finished=True
            feedback_string = 'Gratuliere, du hast alles richtig gemacht!\n\n'
            feedback_string += 'Es geht nun weiter mit dem tatsächlichen Experiment.\n\n'
            feedback_string += 'Bitte drücke eine Taste um fortzufahren.'
            text_feedback.text = feedback_string
        else:
            feedback_string = ''
            if misses:
                feedback_string += f'Du hast {misses}x verpasst eine Taste bei einem auf dem Kopf stehenden Bild zu drücken.\n'
            if false_alarms:
                feedback_string += f'Du hast {false_alarms}x an der falschen  Stelle gedrückt.\n'  
            feedback_string += '\n\nDrücke bitte nur eine Taste, wenn ein Bild falsch herum ist. '
            feedback_string += 'Falls du Fragen hast, stelle diese gerne dem Experimentleiter.\n'
            feedback_string += '\nDrücke eine beliebige Taste um die Übung zu wiederholen.'
            text_feedback.text = feedback_string
        
            # reset counters
            i_localizer = 0
            false_alarms = 0
            misses = 0
            i_block = 0
        # store start times for localizer_feedback
        localizer_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        localizer_feedback.tStart = globalClock.getTime(format='float')
        localizer_feedback.status = STARTED
        thisExp.addData('localizer_feedback.started', localizer_feedback.tStart)
        localizer_feedback.maxDuration = None
        # keep track of which components have finished
        localizer_feedbackComponents = localizer_feedback.components
        for thisComponent in localizer_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "localizer_feedback" ---
        # if trial has changed, end Routine now
        if isinstance(localizer_repetition, data.TrialHandler2) and thisLocalizer_repetition.thisN != localizer_repetition.thisTrial.thisN:
            continueRoutine = False
        localizer_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *key_resp_4* updates
            waitOnFlip = False
            
            # if key_resp_4 is starting this frame...
            if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_4.frameNStart = frameN  # exact frame index
                key_resp_4.tStart = t  # local t and not account for scr refresh
                key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_4.started')
                # update status
                key_resp_4.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_4.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_4.getKeys(keyList=['y', 'g', 'b', 'r'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_4_allKeys.extend(theseKeys)
                if len(_key_resp_4_allKeys):
                    key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                    key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                    key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_feedback* updates
            
            # if text_feedback is starting this frame...
            if text_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_feedback.frameNStart = frameN  # exact frame index
                text_feedback.tStart = t  # local t and not account for scr refresh
                text_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_feedback.started')
                # update status
                text_feedback.status = STARTED
                text_feedback.setAutoDraw(True)
            
            # if text_feedback is active this frame...
            if text_feedback.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                localizer_feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in localizer_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "localizer_feedback" ---
        for thisComponent in localizer_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for localizer_feedback
        localizer_feedback.tStop = globalClock.getTime(format='float')
        localizer_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('localizer_feedback.stopped', localizer_feedback.tStop)
        # check responses
        if key_resp_4.keys in ['', [], None]:  # No response was made
            key_resp_4.keys = None
        localizer_repetition.addData('key_resp_4.keys',key_resp_4.keys)
        if key_resp_4.keys != None:  # we had a response
            localizer_repetition.addData('key_resp_4.rt', key_resp_4.rt)
            localizer_repetition.addData('key_resp_4.duration', key_resp_4.duration)
        # the Routine "localizer_feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 0.0 repeats of 'localizer_repetition'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    blocks = data.TrialHandler2(
        name='blocks',
        nReps=n_block, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(blocks)  # add the loop to the experiment
    thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBlock in blocks:
        currentLoop = blocks
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "counting_code" ---
        # create an object to store info about Routine counting_code
        counting_code = data.Routine(
            name='counting_code',
            components=[],
        )
        counting_code.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_9
        i_localizer = 0
        i_block = blocks.thisN
        # store start times for counting_code
        counting_code.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        counting_code.tStart = globalClock.getTime(format='float')
        counting_code.status = STARTED
        thisExp.addData('counting_code.started', counting_code.tStart)
        counting_code.maxDuration = None
        # keep track of which components have finished
        counting_codeComponents = counting_code.components
        for thisComponent in counting_code.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "counting_code" ---
        # if trial has changed, end Routine now
        if isinstance(blocks, data.TrialHandler2) and thisBlock.thisN != blocks.thisTrial.thisN:
            continueRoutine = False
        counting_code.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                counting_code.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in counting_code.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "counting_code" ---
        for thisComponent in counting_code.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for counting_code
        counting_code.tStop = globalClock.getTime(format='float')
        counting_code.tStopRefresh = tThisFlipGlobal
        thisExp.addData('counting_code.stopped', counting_code.tStop)
        # the Routine "counting_code" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        localizer_trials = data.TrialHandler2(
            name='localizer_trials',
            nReps=n_localizer_trials, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(localizer_trials)  # add the loop to the experiment
        thisLocalizer_trial = localizer_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLocalizer_trial.rgb)
        if thisLocalizer_trial != None:
            for paramName in thisLocalizer_trial:
                globals()[paramName] = thisLocalizer_trial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisLocalizer_trial in localizer_trials:
            currentLoop = localizer_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisLocalizer_trial.rgb)
            if thisLocalizer_trial != None:
                for paramName in thisLocalizer_trial:
                    globals()[paramName] = thisLocalizer_trial[paramName]
            
            # --- Prepare to start Routine "fixation_cross" ---
            # create an object to store info about Routine fixation_cross
            fixation_cross = data.Routine(
                name='fixation_cross',
                components=[localizer_fixation],
            )
            fixation_cross.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for fixation_cross
            fixation_cross.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            fixation_cross.tStart = globalClock.getTime(format='float')
            fixation_cross.status = STARTED
            thisExp.addData('fixation_cross.started', fixation_cross.tStart)
            fixation_cross.maxDuration = None
            # keep track of which components have finished
            fixation_crossComponents = fixation_cross.components
            for thisComponent in fixation_cross.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation_cross" ---
            # if trial has changed, end Routine now
            if isinstance(localizer_trials, data.TrialHandler2) and thisLocalizer_trial.thisN != localizer_trials.thisTrial.thisN:
                continueRoutine = False
            fixation_cross.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *localizer_fixation* updates
                
                # if localizer_fixation is starting this frame...
                if localizer_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    localizer_fixation.frameNStart = frameN  # exact frame index
                    localizer_fixation.tStart = t  # local t and not account for scr refresh
                    localizer_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(localizer_fixation, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'localizer_fixation.started')
                    # update status
                    localizer_fixation.status = STARTED
                    localizer_fixation.setAutoDraw(True)
                
                # if localizer_fixation is active this frame...
                if localizer_fixation.status == STARTED:
                    # update params
                    pass
                
                # if localizer_fixation is stopping this frame...
                if localizer_fixation.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > localizer_fixation.tStartRefresh + t_fixation_cross-frameTolerance:
                        # keep track of stop time/frame for later
                        localizer_fixation.tStop = t  # not accounting for scr refresh
                        localizer_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                        localizer_fixation.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'localizer_fixation.stopped')
                        # update status
                        localizer_fixation.status = FINISHED
                        localizer_fixation.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    fixation_cross.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixation_cross.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation_cross" ---
            for thisComponent in fixation_cross.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for fixation_cross
            fixation_cross.tStop = globalClock.getTime(format='float')
            fixation_cross.tStopRefresh = tThisFlipGlobal
            thisExp.addData('fixation_cross.stopped', fixation_cross.tStop)
            # the Routine "fixation_cross" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "localizer" ---
            # create an object to store info about Routine localizer
            localizer = data.Routine(
                name='localizer',
                components=[localizer_img, localizer_isi, key_resp_localizer, sound_wrong, sound_correct],
            )
            localizer.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_localizer
            key_resp_localizer.keys = []
            key_resp_localizer.rt = []
            _key_resp_localizer_allKeys = []
            # Run 'Begin Routine' code from localizer_code
            df_block = df_localizer[df_localizer['block']==i_block].reset_index(drop = True)
            df_trial = df_block.iloc[i_localizer]
            
            localizer_img.image = "stimuli/" + df_trial.image
            t_isi_localizer = df_trial.iti
            is_distractor = df_trial.distractor
            
            # prevent inactive sound from blocking trial finish
            sound_correct.status = FINISHED
            sound_wrong.status = FINISHED
            played = False
            
            if is_distractor:
                localizer_img.ori  = 180 #image flipped
            else:
                localizer_img.ori  = 0
            
            msg = f'localizer block {i_block}/{n_block} trial {i_localizer}/{len(df_block)}'
            msg += f' {"[FLIPPED]" if is_distractor else ""} isi={t_isi_localizer:.2f}s'
            print(msg)
            
            # set the break conditional to true to display a break
            break_conditional_nReps = int(i_block in breaks_after_block)
            sound_wrong.setSound('stimuli/soundError.wav', secs=0.5, hamming=True)
            sound_wrong.setVolume(1.0, log=False)
            sound_wrong.seek(0)
            sound_correct.setSound('stimuli/soundCoin.wav', secs=0.5, hamming=True)
            sound_correct.setVolume(1.0, log=False)
            sound_correct.seek(0)
            # store start times for localizer
            localizer.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            localizer.tStart = globalClock.getTime(format='float')
            localizer.status = STARTED
            thisExp.addData('localizer.started', localizer.tStart)
            localizer.maxDuration = None
            # keep track of which components have finished
            localizerComponents = localizer.components
            for thisComponent in localizer.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "localizer" ---
            # if trial has changed, end Routine now
            if isinstance(localizer_trials, data.TrialHandler2) and thisLocalizer_trial.thisN != localizer_trials.thisTrial.thisN:
                continueRoutine = False
            localizer.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *localizer_img* updates
                
                # if localizer_img is starting this frame...
                if localizer_img.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    localizer_img.frameNStart = frameN  # exact frame index
                    localizer_img.tStart = t  # local t and not account for scr refresh
                    localizer_img.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(localizer_img, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'localizer_img.started')
                    # update status
                    localizer_img.status = STARTED
                    localizer_img.setAutoDraw(True)
                
                # if localizer_img is active this frame...
                if localizer_img.status == STARTED:
                    # update params
                    pass
                
                # if localizer_img is stopping this frame...
                if localizer_img.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > localizer_img.tStartRefresh + t_img_localizer-frameTolerance:
                        # keep track of stop time/frame for later
                        localizer_img.tStop = t  # not accounting for scr refresh
                        localizer_img.tStopRefresh = tThisFlipGlobal  # on global time
                        localizer_img.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'localizer_img.stopped')
                        # update status
                        localizer_img.status = FINISHED
                        localizer_img.setAutoDraw(False)
                
                # *localizer_isi* updates
                
                # if localizer_isi is starting this frame...
                if localizer_isi.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    localizer_isi.frameNStart = frameN  # exact frame index
                    localizer_isi.tStart = t  # local t and not account for scr refresh
                    localizer_isi.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(localizer_isi, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'localizer_isi.started')
                    # update status
                    localizer_isi.status = STARTED
                    localizer_isi.setAutoDraw(True)
                
                # if localizer_isi is active this frame...
                if localizer_isi.status == STARTED:
                    # update params
                    pass
                
                # if localizer_isi is stopping this frame...
                if localizer_isi.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > localizer_isi.tStartRefresh + t_isi_localizer-frameTolerance:
                        # keep track of stop time/frame for later
                        localizer_isi.tStop = t  # not accounting for scr refresh
                        localizer_isi.tStopRefresh = tThisFlipGlobal  # on global time
                        localizer_isi.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'localizer_isi.stopped')
                        # update status
                        localizer_isi.status = FINISHED
                        localizer_isi.setAutoDraw(False)
                
                # *key_resp_localizer* updates
                waitOnFlip = False
                
                # if key_resp_localizer is starting this frame...
                if key_resp_localizer.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_localizer.frameNStart = frameN  # exact frame index
                    key_resp_localizer.tStart = t  # local t and not account for scr refresh
                    key_resp_localizer.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_localizer, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_localizer.started')
                    # update status
                    key_resp_localizer.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_localizer.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_localizer.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp_localizer is stopping this frame...
                if key_resp_localizer.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_localizer.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_localizer.tStop = t  # not accounting for scr refresh
                        key_resp_localizer.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_localizer.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_localizer.stopped')
                        # update status
                        key_resp_localizer.status = FINISHED
                        key_resp_localizer.status = FINISHED
                if key_resp_localizer.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_localizer.getKeys(keyList=['y', 'g', 'b', 'r'], ignoreKeys=["escape"], waitRelease=True)
                    _key_resp_localizer_allKeys.extend(theseKeys)
                    if len(_key_resp_localizer_allKeys):
                        key_resp_localizer.keys = _key_resp_localizer_allKeys[-1].name  # just the last key pressed
                        key_resp_localizer.rt = _key_resp_localizer_allKeys[-1].rt
                        key_resp_localizer.duration = _key_resp_localizer_allKeys[-1].duration
                # Run 'Each Frame' code from localizer_code
                # play for wrong/correct button presses
                if len(key_resp_localizer.keys) and not played:
                    played = True
                    if is_distractor:
                        sound_correct.play()
                        log('Correct press')
                    else:
                        sound_wrong.play()
                        false_alarms += 1
                        log('False alarm')
                
                # play error in case of missed button press
                if is_distractor and key_resp_localizer.status==FINISHED and not played:
                    played = True
                    sound_wrong.play()
                    misses += 1
                    log('Miss')
                
                
                
                sound_correct.status = FINISHED
                sound_wrong.status = FINISHED
                
                # *sound_wrong* updates
                
                # if sound_wrong is starting this frame...
                if sound_wrong.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    sound_wrong.frameNStart = frameN  # exact frame index
                    sound_wrong.tStart = t  # local t and not account for scr refresh
                    sound_wrong.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('sound_wrong.started', tThisFlipGlobal)
                    # update status
                    sound_wrong.status = STARTED
                    sound_wrong.play(when=win)  # sync with win flip
                
                # if sound_wrong is stopping this frame...
                if sound_wrong.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sound_wrong.tStartRefresh + 0.5-frameTolerance or sound_wrong.isFinished:
                        # keep track of stop time/frame for later
                        sound_wrong.tStop = t  # not accounting for scr refresh
                        sound_wrong.tStopRefresh = tThisFlipGlobal  # on global time
                        sound_wrong.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sound_wrong.stopped')
                        # update status
                        sound_wrong.status = FINISHED
                        sound_wrong.stop()
                
                # *sound_correct* updates
                
                # if sound_correct is starting this frame...
                if sound_correct.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                    # keep track of start time/frame for later
                    sound_correct.frameNStart = frameN  # exact frame index
                    sound_correct.tStart = t  # local t and not account for scr refresh
                    sound_correct.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('sound_correct.started', tThisFlipGlobal)
                    # update status
                    sound_correct.status = STARTED
                    sound_correct.play(when=win)  # sync with win flip
                
                # if sound_correct is stopping this frame...
                if sound_correct.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sound_correct.tStartRefresh + 0.5-frameTolerance or sound_correct.isFinished:
                        # keep track of stop time/frame for later
                        sound_correct.tStop = t  # not accounting for scr refresh
                        sound_correct.tStopRefresh = tThisFlipGlobal  # on global time
                        sound_correct.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sound_correct.stopped')
                        # update status
                        sound_correct.status = FINISHED
                        sound_correct.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[sound_wrong, sound_correct]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    localizer.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in localizer.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "localizer" ---
            for thisComponent in localizer.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for localizer
            localizer.tStop = globalClock.getTime(format='float')
            localizer.tStopRefresh = tThisFlipGlobal
            thisExp.addData('localizer.stopped', localizer.tStop)
            # check responses
            if key_resp_localizer.keys in ['', [], None]:  # No response was made
                key_resp_localizer.keys = None
            localizer_trials.addData('key_resp_localizer.keys',key_resp_localizer.keys)
            if key_resp_localizer.keys != None:  # we had a response
                localizer_trials.addData('key_resp_localizer.rt', key_resp_localizer.rt)
                localizer_trials.addData('key_resp_localizer.duration', key_resp_localizer.duration)
            # Run 'End Routine' code from localizer_code
            i_localizer+=1
            sound_wrong.pause()  # ensure sound has stopped at end of Routine
            sound_correct.pause()  # ensure sound has stopped at end of Routine
            # the Routine "localizer" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed n_localizer_trials repeats of 'localizer_trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # set up handler to look after randomisation of conditions etc
        break_conditional = data.TrialHandler2(
            name='break_conditional',
            nReps=break_conditional_nReps, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(break_conditional)  # add the loop to the experiment
        thisBreak_conditional = break_conditional.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisBreak_conditional.rgb)
        if thisBreak_conditional != None:
            for paramName in thisBreak_conditional:
                globals()[paramName] = thisBreak_conditional[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisBreak_conditional in break_conditional:
            currentLoop = break_conditional
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisBreak_conditional.rgb)
            if thisBreak_conditional != None:
                for paramName in thisBreak_conditional:
                    globals()[paramName] = thisBreak_conditional[paramName]
            
            # --- Prepare to start Routine "break_3" ---
            # create an object to store info about Routine break_3
            break_3 = data.Routine(
                name='break_3',
                components=[text, key_resp_10],
            )
            break_3.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_2
            print('break started')
            if language=='english':
                text.text = f"""<short break>
               
                Please take a short breather.
            
                Press any button if you want to continue with the next block."""
                
            
            # create starting attributes for key_resp_10
            key_resp_10.keys = []
            key_resp_10.rt = []
            _key_resp_10_allKeys = []
            # store start times for break_3
            break_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            break_3.tStart = globalClock.getTime(format='float')
            break_3.status = STARTED
            thisExp.addData('break_3.started', break_3.tStart)
            break_3.maxDuration = None
            # keep track of which components have finished
            break_3Components = break_3.components
            for thisComponent in break_3.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "break_3" ---
            # if trial has changed, end Routine now
            if isinstance(break_conditional, data.TrialHandler2) and thisBreak_conditional.thisN != break_conditional.thisTrial.thisN:
                continueRoutine = False
            break_3.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text* updates
                
                # if text is starting this frame...
                if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text.frameNStart = frameN  # exact frame index
                    text.tStart = t  # local t and not account for scr refresh
                    text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.started')
                    # update status
                    text.status = STARTED
                    text.setAutoDraw(True)
                
                # if text is active this frame...
                if text.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_10* updates
                waitOnFlip = False
                
                # if key_resp_10 is starting this frame...
                if key_resp_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_10.frameNStart = frameN  # exact frame index
                    key_resp_10.tStart = t  # local t and not account for scr refresh
                    key_resp_10.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_10, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_10.started')
                    # update status
                    key_resp_10.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_10.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_10.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_10.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_10.getKeys(keyList=['y','b','r', 'g'], ignoreKeys=["escape"], waitRelease=True)
                    _key_resp_10_allKeys.extend(theseKeys)
                    if len(_key_resp_10_allKeys):
                        key_resp_10.keys = _key_resp_10_allKeys[-1].name  # just the last key pressed
                        key_resp_10.rt = _key_resp_10_allKeys[-1].rt
                        key_resp_10.duration = _key_resp_10_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break_3.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in break_3.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "break_3" ---
            for thisComponent in break_3.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for break_3
            break_3.tStop = globalClock.getTime(format='float')
            break_3.tStopRefresh = tThisFlipGlobal
            thisExp.addData('break_3.stopped', break_3.tStop)
            # Run 'End Routine' code from code_2
            print('break ended')
            core.wait(0.01)  # wait so trigger can reset
            # check responses
            if key_resp_10.keys in ['', [], None]:  # No response was made
                key_resp_10.keys = None
            break_conditional.addData('key_resp_10.keys',key_resp_10.keys)
            if key_resp_10.keys != None:  # we had a response
                break_conditional.addData('key_resp_10.rt', key_resp_10.rt)
                break_conditional.addData('key_resp_10.duration', key_resp_10.duration)
            # the Routine "break_3" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed break_conditional_nReps repeats of 'break_conditional'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        thisExp.nextEntry()
        
    # completed n_block repeats of 'blocks'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "instruct_end" ---
    # create an object to store info about Routine instruct_end
    instruct_end = data.Routine(
        name='instruct_end',
        components=[endtask1],
    )
    instruct_end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for instruct_end
    instruct_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_end.tStart = globalClock.getTime(format='float')
    instruct_end.status = STARTED
    thisExp.addData('instruct_end.started', instruct_end.tStart)
    instruct_end.maxDuration = None
    # keep track of which components have finished
    instruct_endComponents = instruct_end.components
    for thisComponent in instruct_end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_end" ---
    instruct_end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *endtask1* updates
        
        # if endtask1 is starting this frame...
        if endtask1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endtask1.frameNStart = frameN  # exact frame index
            endtask1.tStart = t  # local t and not account for scr refresh
            endtask1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endtask1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'endtask1.started')
            # update status
            endtask1.status = STARTED
            endtask1.setAutoDraw(True)
        
        # if endtask1 is active this frame...
        if endtask1.status == STARTED:
            # update params
            pass
        
        # if endtask1 is stopping this frame...
        if endtask1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > endtask1.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                endtask1.tStop = t  # not accounting for scr refresh
                endtask1.tStopRefresh = tThisFlipGlobal  # on global time
                endtask1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'endtask1.stopped')
                # update status
                endtask1.status = FINISHED
                endtask1.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instruct_end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_end" ---
    for thisComponent in instruct_end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_end
    instruct_end.tStop = globalClock.getTime(format='float')
    instruct_end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruct_end.stopped', instruct_end.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if instruct_end.maxDurationReached:
        routineTimer.addTime(-instruct_end.maxDuration)
    elif instruct_end.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='comma')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
