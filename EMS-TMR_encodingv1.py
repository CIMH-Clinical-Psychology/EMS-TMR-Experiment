#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on März 17, 2025, at 16:43
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
prefs.hardware['audioLatencyMode'] = '3'
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

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'EMS-TMR_encodingv1'  # from the Builder filename that created this script
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
_winSize = [1280, 720]
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
        originPath='C:\\Users\\elena\\Nextcloud\\Masterthesis_EMS-TMR\\EMS-TMR-Experiment\\EMS-TMR_encodingv1.py',
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
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
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
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1.0000, 1.0000, 1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
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
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('key_resp_4') is None:
        # initialise key_resp_4
        key_resp_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_4',
        )
    # create speaker 'encoding_word_prac'
    deviceManager.addDevice(
        deviceName='encoding_word_prac',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_resp_3') is None:
        # initialise key_resp_3
        key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_3',
        )
    # create speaker 'encoding_word'
    deviceManager.addDevice(
        deviceName='encoding_word',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_resp_5') is None:
        # initialise key_resp_5
        key_resp_5 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_5',
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
    t_img_encoding = 0.5
    t_isi_encoding = 2.5
    t_buffer = 10
    image_size = 0.5
    t_fixation_cross = 1.5
    
    n_encoding_trials_prac = 6
    n_encoding_trials = 256
    cross_color = (-1.0000, -1.0000, -1.0000)
    letter_height=0.03
    language = "german"
    
    trigger_img = {}
    trigger_img['dog'] = 1
    trigger_img['car']   = 2
    trigger_img['flower']   = 3
    trigger_img['face']   = 4
    
    trigger_break_start = 91
    trigger_break_stop = 92
    trigger_buf_start = 61
    trigger_buf_stop = 62
    trigger_encoding_sound0 = 30
    trigger_encoding_sound1 = 31
    trigger_fixation_pre1 = 81
    trigger_fixation_pre2 = 82
    
    # these get added depending on the trial
    # ie. Gesicht in localizer = 1, as cue = 11, as sequence=21
    trigger_base_val_encoding = 0
    
    # Run 'Begin Experiment' code from startup
    import pandas as pd
    
    subj_id = expInfo['participant']
    
    df_encoding = pd.read_excel(f"C:/Users/elena/Nextcloud/Masterthesis_EMS-TMR/EMS-TMR-Experiment/sequences/{subj_id}_pairings.xlsx")
    df_encoding_prac = pd.read_excel (f"C:/Users/elena/Nextcloud/Masterthesis_EMS-TMR/EMS-TMR-Experiment/sequences/encoding_practice.xlsx")
    # set variables we will access later
    i_encoding = 0
    i_encoding_prac = 0
    # set the number of repetitions we have
    n_encoding_trials = df_encoding.shape [0]+1
    n_encoding_trials_prac = 3
    
    if PILOTING:
        n_encoding_trials = 5
    
    print('DUMMY TRIALS: subj_id is set to 0')
    
    # set variables we will access later
    i_encoding = 0
    # Run 'Begin Experiment' code from function
    import os.path as osp
    def get_image_name(filename):
        return osp.splitext(osp.basename(filename))[0]
    
    # --- Initialize components for Routine "instructions_part2" ---
    instructions = visual.TextStim(win=win, name='instructions',
        text='Willkommen bei der zweiten Aufgabe des Experiments.\n\nHier werden dir Word-Bild Paare präsentiert, wobei dir das Word vorgespielt und das Bild im Anschluss visuell präsentiert wird. Deine Aufgabe ist es, sich die Word-Bild Paare bestmöglichst zu merken.  Versuche aufmerksam zu bleiben! Deine Lernleistung wird im Anschluss abgefragt werden.\n\nDrücke eine beliebige Taste um fortzufahren.',
        font='Times New Roman',
        pos=(0, 0), draggable=False, height=letter_height, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "instructions_part2_1" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='Um sich die Word-Bild Paare besser merken zu können stelle dir bitte ein Bild vor, welches sowohl das Word, als auch das Bild enthält.\n\nBeispiel:\n\n\n\n\n\n\n\n\n\n\n\n\nDrücke nun eine beliebige Taste, um mit einen Beispieldurchlauf zu starten.',
        font='Times New Roman',
        pos=(0, 0), draggable=False, height=letter_height, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='stimuli/mentales Bild.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.0), draggable=False, size=(0.8, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    # Run 'Begin Experiment' code from code_7
    if language == "english":
        instructions = """ In order to memorize the word-picture pairs better, please imagine a mental picture that contains both the word and the picture.
        Now press any key to start a practice run."""
    
    # --- Initialize components for Routine "fixation_cross" ---
    cross = visual.TextStim(win=win, name='cross',
        text='X',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "word_stimuli_prac" ---
    encoding_word_prac = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker='encoding_word_prac',    name='encoding_word_prac'
    )
    encoding_word_prac.setVolume(1.0)
    
    # --- Initialize components for Routine "image_stimuli_prac_2" ---
    encoding_img_prac = visual.ImageStim(
        win=win,
        name='encoding_img_prac', 
        image='stimuli/practice_encoding/dogxxx.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "instructions_part2_2" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Das waren jetzt 3 Beispieldurchgänge. Konntest du dir zu jedem Paar ein mentales Bild bilden?\n\nDie soeben gezeigten Wort-Bild Paaare gehören NICHT zu den Paaren, die du dir merken sollst.\n\nIm Folgenden werden dir nun alle Wort-Bild Paare gezeigt, von denen du dir so viele wie möglich merken sollst.\n\nSobald du bereit bist können wir mit dem Experiment starten. Drücke eine beliebige Taste um zu starten.',
        font='Times New Roman',
        pos=(0, 0), draggable=False, height=letter_height, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard(deviceName='key_resp_3')
    
    # --- Initialize components for Routine "fixation_cross" ---
    cross = visual.TextStim(win=win, name='cross',
        text='X',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "word_stimuli" ---
    encoding_word = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker='encoding_word',    name='encoding_word'
    )
    encoding_word.setVolume(1.0)
    
    # --- Initialize components for Routine "image_stimuli" ---
    encoding_img = visual.ImageStim(
        win=win,
        name='encoding_img', 
        image='stimuli/dog/dog001.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "break_2" ---
    # Run 'Begin Experiment' code from code_10
    if i_encoding != (29,59,89):
        continueRoutine = False
    key_resp_5 = keyboard.Keyboard(deviceName='key_resp_5')
    text_4 = visual.TextStim(win=win, name='text_4',
        text='Eine kurze Pause.\n\nBitte nimm eine kurze Verschnaufpause. Sobald du dich wieder bereit fühlst, drücke bitte eine beliebige Taste, um mit der Aufgabe fortzufahren.',
        font='Times New Roman',
        pos=(0, 0), draggable=False, height=letter_height, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "instruction_end" ---
    text = visual.TextStim(win=win, name='text',
        text='Super, du hast nun alle 120 Word-Bild Paare gesehen und versucht, dir so viele wie möglich zu merken, indem du dir jeweils ein mentales Bild des Wortes und des Bildes gebildet hast.\n\nNun geht es weiter mit der nächsten Aufgabe, um das Gelernte noch weiter zu vertiefen. ',
        font='Times New Roman',
        pos=(0, 0), draggable=False, height=letter_height, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
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
    
    # --- Prepare to start Routine "instructions_part2" ---
    # create an object to store info about Routine instructions_part2
    instructions_part2 = data.Routine(
        name='instructions_part2',
        components=[instructions, key_resp],
    )
    instructions_part2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # Run 'Begin Routine' code from code
    if language == "english":
        instructions = """ Welcome to the second task of the experiment.
        Here you will be presented with word-image pairs, whereby the word is played to you and the image is then presented visually. Your task is to memorize the word-picture pairs as best you can.  Try to stay attentive! Your learning performance will be tested afterwards.
       Press any key to continue."""
    # store start times for instructions_part2
    instructions_part2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_part2.tStart = globalClock.getTime(format='float')
    instructions_part2.status = STARTED
    thisExp.addData('instructions_part2.started', instructions_part2.tStart)
    instructions_part2.maxDuration = None
    # keep track of which components have finished
    instructions_part2Components = instructions_part2.components
    for thisComponent in instructions_part2.components:
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
    
    # --- Run Routine "instructions_part2" ---
    instructions_part2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 60.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions* updates
        
        # if instructions is starting this frame...
        if instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions.frameNStart = frameN  # exact frame index
            instructions.tStart = t  # local t and not account for scr refresh
            instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions.started')
            # update status
            instructions.status = STARTED
            instructions.setAutoDraw(True)
        
        # if instructions is active this frame...
        if instructions.status == STARTED:
            # update params
            pass
        
        # if instructions is stopping this frame...
        if instructions.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > instructions.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                instructions.tStop = t  # not accounting for scr refresh
                instructions.tStopRefresh = tThisFlipGlobal  # on global time
                instructions.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instructions.stopped')
                # update status
                instructions.status = FINISHED
                instructions.setAutoDraw(False)
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp is stopping this frame...
        if key_resp.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                key_resp.tStop = t  # not accounting for scr refresh
                key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                key_resp.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.stopped')
                # update status
                key_resp.status = FINISHED
                key_resp.status = FINISHED
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['y','b','r', 'g'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
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
            instructions_part2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_part2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_part2" ---
    for thisComponent in instructions_part2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_part2
    instructions_part2.tStop = globalClock.getTime(format='float')
    instructions_part2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_part2.stopped', instructions_part2.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if instructions_part2.maxDurationReached:
        routineTimer.addTime(-instructions_part2.maxDuration)
    elif instructions_part2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-60.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "instructions_part2_1" ---
    # create an object to store info about Routine instructions_part2_1
    instructions_part2_1 = data.Routine(
        name='instructions_part2_1',
        components=[text_3, image_2, key_resp_4],
    )
    instructions_part2_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_4
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # store start times for instructions_part2_1
    instructions_part2_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_part2_1.tStart = globalClock.getTime(format='float')
    instructions_part2_1.status = STARTED
    thisExp.addData('instructions_part2_1.started', instructions_part2_1.tStart)
    instructions_part2_1.maxDuration = None
    # keep track of which components have finished
    instructions_part2_1Components = instructions_part2_1.components
    for thisComponent in instructions_part2_1.components:
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
    
    # --- Run Routine "instructions_part2_1" ---
    instructions_part2_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 60.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # if text_3 is stopping this frame...
        if text_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_3.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                text_3.tStop = t  # not accounting for scr refresh
                text_3.tStopRefresh = tThisFlipGlobal  # on global time
                text_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_3.stopped')
                # update status
                text_3.status = FINISHED
                text_3.setAutoDraw(False)
        
        # *image_2* updates
        
        # if image_2 is starting this frame...
        if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_2.frameNStart = frameN  # exact frame index
            image_2.tStart = t  # local t and not account for scr refresh
            image_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_2.started')
            # update status
            image_2.status = STARTED
            image_2.setAutoDraw(True)
        
        # if image_2 is active this frame...
        if image_2.status == STARTED:
            # update params
            pass
        
        # if image_2 is stopping this frame...
        if image_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_2.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                image_2.tStop = t  # not accounting for scr refresh
                image_2.tStopRefresh = tThisFlipGlobal  # on global time
                image_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_2.stopped')
                # update status
                image_2.status = FINISHED
                image_2.setAutoDraw(False)
        
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
        
        # if key_resp_4 is stopping this frame...
        if key_resp_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp_4.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                key_resp_4.tStop = t  # not accounting for scr refresh
                key_resp_4.tStopRefresh = tThisFlipGlobal  # on global time
                key_resp_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_4.stopped')
                # update status
                key_resp_4.status = FINISHED
                key_resp_4.status = FINISHED
        if key_resp_4.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_4.getKeys(keyList=['y','b','r', 'g'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
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
            instructions_part2_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_part2_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_part2_1" ---
    for thisComponent in instructions_part2_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_part2_1
    instructions_part2_1.tStop = globalClock.getTime(format='float')
    instructions_part2_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_part2_1.stopped', instructions_part2_1.tStop)
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if instructions_part2_1.maxDurationReached:
        routineTimer.addTime(-instructions_part2_1.maxDuration)
    elif instructions_part2_1.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-60.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    encoding_trials_prac = data.TrialHandler2(
        name='encoding_trials_prac',
        nReps=n_encoding_trials_prac, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(encoding_trials_prac)  # add the loop to the experiment
    thisEncoding_trials_prac = encoding_trials_prac.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisEncoding_trials_prac.rgb)
    if thisEncoding_trials_prac != None:
        for paramName in thisEncoding_trials_prac:
            globals()[paramName] = thisEncoding_trials_prac[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisEncoding_trials_prac in encoding_trials_prac:
        currentLoop = encoding_trials_prac
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisEncoding_trials_prac.rgb)
        if thisEncoding_trials_prac != None:
            for paramName in thisEncoding_trials_prac:
                globals()[paramName] = thisEncoding_trials_prac[paramName]
        
        # --- Prepare to start Routine "fixation_cross" ---
        # create an object to store info about Routine fixation_cross
        fixation_cross = data.Routine(
            name='fixation_cross',
            components=[cross],
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
        if isinstance(encoding_trials_prac, data.TrialHandler2) and thisEncoding_trials_prac.thisN != encoding_trials_prac.thisTrial.thisN:
            continueRoutine = False
        fixation_cross.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross* updates
            
            # if cross is starting this frame...
            if cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross.frameNStart = frameN  # exact frame index
                cross.tStart = t  # local t and not account for scr refresh
                cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross.started')
                # update status
                cross.status = STARTED
                cross.setAutoDraw(True)
            
            # if cross is active this frame...
            if cross.status == STARTED:
                # update params
                pass
            
            # if cross is stopping this frame...
            if cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    cross.tStop = t  # not accounting for scr refresh
                    cross.tStopRefresh = tThisFlipGlobal  # on global time
                    cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross.stopped')
                    # update status
                    cross.status = FINISHED
                    cross.setAutoDraw(False)
            
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
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation_cross.maxDurationReached:
            routineTimer.addTime(-fixation_cross.maxDuration)
        elif fixation_cross.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        
        # --- Prepare to start Routine "word_stimuli_prac" ---
        # create an object to store info about Routine word_stimuli_prac
        word_stimuli_prac = data.Routine(
            name='word_stimuli_prac',
            components=[encoding_word_prac],
        )
        word_stimuli_prac.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_8
        df_trial_prac = df_encoding_prac.iloc[i_encoding]
        print (df_trial_prac)
        
        
        encoding_word_prac.wav = "stimuli/practice_encoding" + df_trial_prac.word
        encoding_word_prac.setSound('A', secs=1.0, hamming=True)
        encoding_word_prac.setVolume(1.0, log=False)
        encoding_word_prac.seek(0)
        # store start times for word_stimuli_prac
        word_stimuli_prac.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        word_stimuli_prac.tStart = globalClock.getTime(format='float')
        word_stimuli_prac.status = STARTED
        thisExp.addData('word_stimuli_prac.started', word_stimuli_prac.tStart)
        word_stimuli_prac.maxDuration = None
        # keep track of which components have finished
        word_stimuli_pracComponents = word_stimuli_prac.components
        for thisComponent in word_stimuli_prac.components:
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
        
        # --- Run Routine "word_stimuli_prac" ---
        # if trial has changed, end Routine now
        if isinstance(encoding_trials_prac, data.TrialHandler2) and thisEncoding_trials_prac.thisN != encoding_trials_prac.thisTrial.thisN:
            continueRoutine = False
        word_stimuli_prac.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *encoding_word_prac* updates
            
            # if encoding_word_prac is starting this frame...
            if encoding_word_prac.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                encoding_word_prac.frameNStart = frameN  # exact frame index
                encoding_word_prac.tStart = t  # local t and not account for scr refresh
                encoding_word_prac.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('encoding_word_prac.started', tThisFlipGlobal)
                # update status
                encoding_word_prac.status = STARTED
                encoding_word_prac.play(when=win)  # sync with win flip
            
            # if encoding_word_prac is stopping this frame...
            if encoding_word_prac.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > encoding_word_prac.tStartRefresh + 1.0-frameTolerance or encoding_word_prac.isFinished:
                    # keep track of stop time/frame for later
                    encoding_word_prac.tStop = t  # not accounting for scr refresh
                    encoding_word_prac.tStopRefresh = tThisFlipGlobal  # on global time
                    encoding_word_prac.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'encoding_word_prac.stopped')
                    # update status
                    encoding_word_prac.status = FINISHED
                    encoding_word_prac.stop()
            
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
                    playbackComponents=[encoding_word_prac]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                word_stimuli_prac.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in word_stimuli_prac.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "word_stimuli_prac" ---
        for thisComponent in word_stimuli_prac.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for word_stimuli_prac
        word_stimuli_prac.tStop = globalClock.getTime(format='float')
        word_stimuli_prac.tStopRefresh = tThisFlipGlobal
        thisExp.addData('word_stimuli_prac.stopped', word_stimuli_prac.tStop)
        encoding_word_prac.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if word_stimuli_prac.maxDurationReached:
            routineTimer.addTime(-word_stimuli_prac.maxDuration)
        elif word_stimuli_prac.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "image_stimuli_prac_2" ---
        # create an object to store info about Routine image_stimuli_prac_2
        image_stimuli_prac_2 = data.Routine(
            name='image_stimuli_prac_2',
            components=[encoding_img_prac],
        )
        image_stimuli_prac_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_9
        df_trial_prac = df_encoding_prac.iloc[i_encoding_prac]
        print (df_trial_prac)
        
        
        encoding_img_prac.image = "stimuli/" + df_trial_prac.image
        # store start times for image_stimuli_prac_2
        image_stimuli_prac_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        image_stimuli_prac_2.tStart = globalClock.getTime(format='float')
        image_stimuli_prac_2.status = STARTED
        thisExp.addData('image_stimuli_prac_2.started', image_stimuli_prac_2.tStart)
        image_stimuli_prac_2.maxDuration = None
        # keep track of which components have finished
        image_stimuli_prac_2Components = image_stimuli_prac_2.components
        for thisComponent in image_stimuli_prac_2.components:
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
        
        # --- Run Routine "image_stimuli_prac_2" ---
        # if trial has changed, end Routine now
        if isinstance(encoding_trials_prac, data.TrialHandler2) and thisEncoding_trials_prac.thisN != encoding_trials_prac.thisTrial.thisN:
            continueRoutine = False
        image_stimuli_prac_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *encoding_img_prac* updates
            
            # if encoding_img_prac is starting this frame...
            if encoding_img_prac.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                encoding_img_prac.frameNStart = frameN  # exact frame index
                encoding_img_prac.tStart = t  # local t and not account for scr refresh
                encoding_img_prac.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(encoding_img_prac, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'encoding_img_prac.started')
                # update status
                encoding_img_prac.status = STARTED
                encoding_img_prac.setAutoDraw(True)
            
            # if encoding_img_prac is active this frame...
            if encoding_img_prac.status == STARTED:
                # update params
                pass
            
            # if encoding_img_prac is stopping this frame...
            if encoding_img_prac.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > encoding_img_prac.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    encoding_img_prac.tStop = t  # not accounting for scr refresh
                    encoding_img_prac.tStopRefresh = tThisFlipGlobal  # on global time
                    encoding_img_prac.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'encoding_img_prac.stopped')
                    # update status
                    encoding_img_prac.status = FINISHED
                    encoding_img_prac.setAutoDraw(False)
            
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
                image_stimuli_prac_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in image_stimuli_prac_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "image_stimuli_prac_2" ---
        for thisComponent in image_stimuli_prac_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for image_stimuli_prac_2
        image_stimuli_prac_2.tStop = globalClock.getTime(format='float')
        image_stimuli_prac_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('image_stimuli_prac_2.stopped', image_stimuli_prac_2.tStop)
        # Run 'End Routine' code from code_9
        i_encoding_prac += 1
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if image_stimuli_prac_2.maxDurationReached:
            routineTimer.addTime(-image_stimuli_prac_2.maxDuration)
        elif image_stimuli_prac_2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        thisExp.nextEntry()
        
    # completed n_encoding_trials_prac repeats of 'encoding_trials_prac'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "instructions_part2_2" ---
    # create an object to store info about Routine instructions_part2_2
    instructions_part2_2 = data.Routine(
        name='instructions_part2_2',
        components=[text_2, key_resp_3],
    )
    instructions_part2_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_3
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # Run 'Begin Routine' code from code_3
    if language == "english":
        text_2 = """These were 3 sample runs. Were you able to form a mental image for each shown pair?
        The word-picture pairs just shown are NOT among the pairs that you should memorize.
        You will now be shown all the word-picture pairs, of which you should memorize as many as possible.
        Press any button to start the experiment."""
    # store start times for instructions_part2_2
    instructions_part2_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_part2_2.tStart = globalClock.getTime(format='float')
    instructions_part2_2.status = STARTED
    thisExp.addData('instructions_part2_2.started', instructions_part2_2.tStart)
    instructions_part2_2.maxDuration = None
    # keep track of which components have finished
    instructions_part2_2Components = instructions_part2_2.components
    for thisComponent in instructions_part2_2.components:
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
    
    # --- Run Routine "instructions_part2_2" ---
    instructions_part2_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 60.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
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
            instructions_part2_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_part2_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_part2_2" ---
    for thisComponent in instructions_part2_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_part2_2
    instructions_part2_2.tStop = globalClock.getTime(format='float')
    instructions_part2_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_part2_2.stopped', instructions_part2_2.tStop)
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if instructions_part2_2.maxDurationReached:
        routineTimer.addTime(-instructions_part2_2.maxDuration)
    elif instructions_part2_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-60.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    encoding_trials = data.TrialHandler2(
        name='encoding_trials',
        nReps=n_encoding_trials, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(encoding_trials)  # add the loop to the experiment
    thisEncoding_trial = encoding_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisEncoding_trial.rgb)
    if thisEncoding_trial != None:
        for paramName in thisEncoding_trial:
            globals()[paramName] = thisEncoding_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisEncoding_trial in encoding_trials:
        currentLoop = encoding_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisEncoding_trial.rgb)
        if thisEncoding_trial != None:
            for paramName in thisEncoding_trial:
                globals()[paramName] = thisEncoding_trial[paramName]
        
        # --- Prepare to start Routine "fixation_cross" ---
        # create an object to store info about Routine fixation_cross
        fixation_cross = data.Routine(
            name='fixation_cross',
            components=[cross],
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
        if isinstance(encoding_trials, data.TrialHandler2) and thisEncoding_trial.thisN != encoding_trials.thisTrial.thisN:
            continueRoutine = False
        fixation_cross.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross* updates
            
            # if cross is starting this frame...
            if cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross.frameNStart = frameN  # exact frame index
                cross.tStart = t  # local t and not account for scr refresh
                cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross.started')
                # update status
                cross.status = STARTED
                cross.setAutoDraw(True)
            
            # if cross is active this frame...
            if cross.status == STARTED:
                # update params
                pass
            
            # if cross is stopping this frame...
            if cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    cross.tStop = t  # not accounting for scr refresh
                    cross.tStopRefresh = tThisFlipGlobal  # on global time
                    cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross.stopped')
                    # update status
                    cross.status = FINISHED
                    cross.setAutoDraw(False)
            
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
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation_cross.maxDurationReached:
            routineTimer.addTime(-fixation_cross.maxDuration)
        elif fixation_cross.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        
        # --- Prepare to start Routine "word_stimuli" ---
        # create an object to store info about Routine word_stimuli
        word_stimuli = data.Routine(
            name='word_stimuli',
            components=[encoding_word],
        )
        word_stimuli.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        encoding_word.setSound('A', secs=1.0, hamming=True)
        encoding_word.setVolume(1.0, log=False)
        encoding_word.seek(0)
        # Run 'Begin Routine' code from code_6
        df_trial = df_encoding.iloc[i_encoding]
        print (df_trial)
        
        
        encoding_word.sound = "stimuli/sounds/de_" + df_trial.word
        # store start times for word_stimuli
        word_stimuli.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        word_stimuli.tStart = globalClock.getTime(format='float')
        word_stimuli.status = STARTED
        thisExp.addData('word_stimuli.started', word_stimuli.tStart)
        word_stimuli.maxDuration = None
        # keep track of which components have finished
        word_stimuliComponents = word_stimuli.components
        for thisComponent in word_stimuli.components:
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
        
        # --- Run Routine "word_stimuli" ---
        # if trial has changed, end Routine now
        if isinstance(encoding_trials, data.TrialHandler2) and thisEncoding_trial.thisN != encoding_trials.thisTrial.thisN:
            continueRoutine = False
        word_stimuli.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *encoding_word* updates
            
            # if encoding_word is starting this frame...
            if encoding_word.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                encoding_word.frameNStart = frameN  # exact frame index
                encoding_word.tStart = t  # local t and not account for scr refresh
                encoding_word.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('encoding_word.started', tThisFlipGlobal)
                # update status
                encoding_word.status = STARTED
                encoding_word.play(when=win)  # sync with win flip
            
            # if encoding_word is stopping this frame...
            if encoding_word.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > encoding_word.tStartRefresh + 1.0-frameTolerance or encoding_word.isFinished:
                    # keep track of stop time/frame for later
                    encoding_word.tStop = t  # not accounting for scr refresh
                    encoding_word.tStopRefresh = tThisFlipGlobal  # on global time
                    encoding_word.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'encoding_word.stopped')
                    # update status
                    encoding_word.status = FINISHED
                    encoding_word.stop()
            
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
                    playbackComponents=[encoding_word]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                word_stimuli.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in word_stimuli.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "word_stimuli" ---
        for thisComponent in word_stimuli.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for word_stimuli
        word_stimuli.tStop = globalClock.getTime(format='float')
        word_stimuli.tStopRefresh = tThisFlipGlobal
        thisExp.addData('word_stimuli.stopped', word_stimuli.tStop)
        encoding_word.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if word_stimuli.maxDurationReached:
            routineTimer.addTime(-word_stimuli.maxDuration)
        elif word_stimuli.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "image_stimuli" ---
        # create an object to store info about Routine image_stimuli
        image_stimuli = data.Routine(
            name='image_stimuli',
            components=[encoding_img],
        )
        image_stimuli.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_5
        df_trial = df_encoding.iloc[i_encoding]
        print (df_trial)
        
        
        encoding_img.image = "stimuli/" + df_trial.image
        
        # store start times for image_stimuli
        image_stimuli.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        image_stimuli.tStart = globalClock.getTime(format='float')
        image_stimuli.status = STARTED
        thisExp.addData('image_stimuli.started', image_stimuli.tStart)
        image_stimuli.maxDuration = None
        # keep track of which components have finished
        image_stimuliComponents = image_stimuli.components
        for thisComponent in image_stimuli.components:
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
        
        # --- Run Routine "image_stimuli" ---
        # if trial has changed, end Routine now
        if isinstance(encoding_trials, data.TrialHandler2) and thisEncoding_trial.thisN != encoding_trials.thisTrial.thisN:
            continueRoutine = False
        image_stimuli.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *encoding_img* updates
            
            # if encoding_img is starting this frame...
            if encoding_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                encoding_img.frameNStart = frameN  # exact frame index
                encoding_img.tStart = t  # local t and not account for scr refresh
                encoding_img.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(encoding_img, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'encoding_img.started')
                # update status
                encoding_img.status = STARTED
                encoding_img.setAutoDraw(True)
            
            # if encoding_img is active this frame...
            if encoding_img.status == STARTED:
                # update params
                pass
            
            # if encoding_img is stopping this frame...
            if encoding_img.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > encoding_img.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    encoding_img.tStop = t  # not accounting for scr refresh
                    encoding_img.tStopRefresh = tThisFlipGlobal  # on global time
                    encoding_img.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'encoding_img.stopped')
                    # update status
                    encoding_img.status = FINISHED
                    encoding_img.setAutoDraw(False)
            
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
                image_stimuli.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in image_stimuli.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "image_stimuli" ---
        for thisComponent in image_stimuli.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for image_stimuli
        image_stimuli.tStop = globalClock.getTime(format='float')
        image_stimuli.tStopRefresh = tThisFlipGlobal
        thisExp.addData('image_stimuli.stopped', image_stimuli.tStop)
        # Run 'End Routine' code from code_5
        i_encoding+=1
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if image_stimuli.maxDurationReached:
            routineTimer.addTime(-image_stimuli.maxDuration)
        elif image_stimuli.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        
        # --- Prepare to start Routine "break_2" ---
        # create an object to store info about Routine break_2
        break_2 = data.Routine(
            name='break_2',
            components=[key_resp_5, text_4],
        )
        break_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_10
        print ("start break")
        if language == "english":
            text_4.text = """<short break>
            Please take a short breather.
            Press any button if you want to continue with the next block."""
        # create starting attributes for key_resp_5
        key_resp_5.keys = []
        key_resp_5.rt = []
        _key_resp_5_allKeys = []
        # store start times for break_2
        break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        break_2.tStart = globalClock.getTime(format='float')
        break_2.status = STARTED
        thisExp.addData('break_2.started', break_2.tStart)
        break_2.maxDuration = None
        # keep track of which components have finished
        break_2Components = break_2.components
        for thisComponent in break_2.components:
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
        
        # --- Run Routine "break_2" ---
        # if trial has changed, end Routine now
        if isinstance(encoding_trials, data.TrialHandler2) and thisEncoding_trial.thisN != encoding_trials.thisTrial.thisN:
            continueRoutine = False
        break_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *key_resp_5* updates
            waitOnFlip = False
            
            # if key_resp_5 is starting this frame...
            if key_resp_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_5.frameNStart = frameN  # exact frame index
                key_resp_5.tStart = t  # local t and not account for scr refresh
                key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_5.started')
                # update status
                key_resp_5.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_5.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_5.getKeys(keyList=['y','b','r', 'g'], ignoreKeys=["escape"], waitRelease=True)
                _key_resp_5_allKeys.extend(theseKeys)
                if len(_key_resp_5_allKeys):
                    key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                    key_resp_5.rt = _key_resp_5_allKeys[-1].rt
                    key_resp_5.duration = _key_resp_5_allKeys[-1].duration
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
                break_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in break_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_2" ---
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for break_2
        break_2.tStop = globalClock.getTime(format='float')
        break_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('break_2.stopped', break_2.tStop)
        # Run 'End Routine' code from code_10
        print ("break ended")
        core.wait(0.01)  # wait so trigger can reset
        # check responses
        if key_resp_5.keys in ['', [], None]:  # No response was made
            key_resp_5.keys = None
        encoding_trials.addData('key_resp_5.keys',key_resp_5.keys)
        if key_resp_5.keys != None:  # we had a response
            encoding_trials.addData('key_resp_5.rt', key_resp_5.rt)
            encoding_trials.addData('key_resp_5.duration', key_resp_5.duration)
        # the Routine "break_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed n_encoding_trials repeats of 'encoding_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "instruction_end" ---
    # create an object to store info about Routine instruction_end
    instruction_end = data.Routine(
        name='instruction_end',
        components=[text],
    )
    instruction_end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_4
    if language == "english":
        text = """Great, you have now seen all 120 word-image pairs and tried to memorize as many as possible by forming a mental image of each word and image.
        Now move on to the next task to consolidate what you have learned. """
    # store start times for instruction_end
    instruction_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruction_end.tStart = globalClock.getTime(format='float')
    instruction_end.status = STARTED
    thisExp.addData('instruction_end.started', instruction_end.tStart)
    instruction_end.maxDuration = None
    # keep track of which components have finished
    instruction_endComponents = instruction_end.components
    for thisComponent in instruction_end.components:
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
    
    # --- Run Routine "instruction_end" ---
    instruction_end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 60.0:
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
        
        # if text is stopping this frame...
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.tStopRefresh = tThisFlipGlobal  # on global time
                text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.stopped')
                # update status
                text.status = FINISHED
                text.setAutoDraw(False)
        
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
            instruction_end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruction_end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction_end" ---
    for thisComponent in instruction_end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruction_end
    instruction_end.tStop = globalClock.getTime(format='float')
    instruction_end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruction_end.stopped', instruction_end.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if instruction_end.maxDurationReached:
        routineTimer.addTime(-instruction_end.maxDuration)
    elif instruction_end.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-60.000000)
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
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
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
