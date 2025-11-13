#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==========================================================================
#         ____                   __       _          _____ ____ ____
#        |  _ \  ___  ___ _ __  / _| __ _| | _____  | ____/ ___/ ___|
#        | | | |/ _ \/ _ \ '_ \| |_ / _` | |/ / _ \ |  _|| |  | |  _
#        | |_| |  __/  __/ |_) |  _| (_| |   <  __/ | |__| |__| |_| |
#        |____/ \___|\___| .__/|_|  \__,_|_|\_\___| |_____\____\____|
#                        |_|
#
#                       --- Deepfake ECG Generator ---
#                https://github.com/vlbthambawita/deepfake-ecg
# ==========================================================================
#
# DeepfakeECG GUI Application
# Copyright (C) 2023-2025 by Vajira Thambawita
# Copyright (C) 2025 by Thomas Dreibholz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Contact:
# * Vajira Thambawita <vajira@simula.no>
# * Thomas Dreibholz <dreibh@simula.no>

import datetime
import deepfakeecg
import ecg_plot
import getopt
import gradio
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import neurokit2
import numpy
import pathlib
import random
import sys
import tempfile
import threading
import torch
import typing
import version
import PIL
import PIL.Image

from typing import Any, Final


# ###### Print log message ##################################################
def log(logstring : str) -> None:
   print(('\x1b[34m' + datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f') +
          ': ' + logstring + '\x1b[0m'));



# ###### DeepFakeECG Plus Session (session with web browser) ################
class Session:

   # ###### Constructor #####################################################
   def __init__(self) -> None:
      self.Lock = threading.Lock()
      self.Counter       : int                             = 0
      self.Selected      : int                             = 0
      self.Results       : list[Any]                       = [ ]
      self.Analysis      : matplotlib.figure.Figure | None = None
      self.Type          : int | None                      = None
      self.TempDirectory : tempfile.TemporaryDirectory     = \
         tempfile.TemporaryDirectory(dir = TempDirectory.name)
      log(f'Prepared temporary directory {self.TempDirectory.name}')

   # ###### Destructor ######################################################
   def __del__(self) -> None:
      log(f'Cleaning up temporary directory {self.TempDirectory.name}')
      self.TempDirectory.cleanup()


TempDirectory : tempfile.TemporaryDirectory[Any]
Sessions      : dict[str,Session] = { }


# ###### Initialize a new session ###########################################
def initializeSession(request: gradio.Request) -> None:
   Sessions[request.session_hash] = Session()
   log(f'Session "{request.session_hash}" initialized => {len(Sessions)} active sessions')


# ###### Clean up a session #################################################
def cleanUpSession(request: gradio.Request) -> None:
   if request.session_hash in Sessions:
      if Sessions[request.session_hash].Analysis:
         matplotlib.pyplot.close(Sessions[request.session_hash].Analysis)
      del Sessions[request.session_hash]
   log(f'Session "{request.session_hash}" cleaned up => {len(Sessions)} active sessions')


# ###### Generate ECGs ######################################################
def predict(numberOfECGs:         int = 1,
            # ecgLengthInSeconds: int = 10,
            ecgTypeString:        str = 'ECG-12',
            generatorModel:       str = 'Default',
            request:              gradio.Request = None) -> tuple[list[tuple[PIL.Image.Image,str]],matplotlib.figure.Figure]:

   ecgLengthInSeconds = 10

   log(f'Session "{request.session_hash}": Generate EGCs!')


   # ====== Set ECG type ====================================================
   ecgType = deepfakeecg.DATA_ECG12
   if ecgTypeString == 'ECG-8':
      ecgType = deepfakeecg.DATA_ECG8
   elif ecgTypeString == 'ECG-12':
      ecgType = deepfakeecg.DATA_ECG12
   else:
      sys.stderr.write(f'WARNING: Invalid ecgTypeString {ecgTypeString}, using ECG-12!\n')

   # ====== Raise Locator.MAXTICKS, if necessary ============================
   matplotlib.ticker.Locator.MAXTICKS = \
       max(1000, ecgLengthInSeconds * deepfakeecg.ECG_SAMPLING_RATE)
   # print(matplotlib.ticker.Locator.MAXTICKS)

   # ====== Generate the ECGs ===============================================
   Sessions[request.session_hash].Results = \
      deepfakeecg.generateDeepfakeECGs(numberOfECGs,
                                       ecgType            = ecgType,
                                       ecgLengthInSeconds = ecgLengthInSeconds,
                                       ecgScaleFactor     = deepfakeecg.ECG_DEFAULT_SCALE_FACTOR,
                                       outputFormat       = deepfakeecg.OUTPUT_TENSOR,
                                       showProgress       = False,
                                       runOnDevice        = runOnDevice)
   Sessions[request.session_hash].Type = ecgType

   # ====== Create a list of image/label tuples for gradio.Gallery ==========
   plotList  : list[tuple[PIL.Image.Image,str]] = [ ]
   ecgNumber : int                              = 1
   info      : Final[str]                       = '25 mm/sec, 1 mV/10 mm'
   for result in Sessions[request.session_hash].Results:

      # ====== Plot ECG =====================================================
      # 1. Convert to NumPy
      # 2. Remove the Timestamp column (0)
      # 3. Convert from µV to mV
      result = result.t().detach().cpu().numpy()[1:] / 1000
      # print(result)

      # ------ ECG-12 -------------------------------------------------------
      if ecgType == deepfakeecg.DATA_ECG12:
         ecg_plot.plot(result,
                       title       = 'ECG-12 – ' + info,
                       sample_rate = deepfakeecg.ECG_SAMPLING_RATE,
                       lead_index  = [ 'I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'III', 'aVR', 'aVL', 'aVF' ],
                       lead_order  = [0, 1, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7],
                       show_grid   = True)
      # ------ ECG-8 --------------------------------------------------------
      else:
         ecg_plot.plot(result,
                       title       = 'ECG-8 – ' + info,
                       sample_rate = deepfakeecg.ECG_SAMPLING_RATE,
                       lead_index  = [ 'I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6' ],
                       lead_order  = [0, 1, 2, 3, 4, 5, 6, 7],
                       show_grid   = True)

      # ====== Generate WebP output =========================================
      imageBuffer = io.BytesIO()
      plt.savefig(imageBuffer, format = 'webp')
      plt.close()
      image : PIL.Image.Image = PIL.Image.open(imageBuffer)
      plotList.append( (image, f'ECG Number {ecgNumber}') )

      ecgNumber = ecgNumber + 1

   # ====== Prepare analysis results for first ECG ==========================
   Sessions[request.session_hash].Analysis = \
      plotAnalysis(Sessions[request.session_hash].Results[0])

   return (plotList, Sessions[request.session_hash].Analysis)


# ###### Plot the analysis ##################################################
def plotAnalysis(data : torch.Tensor) -> matplotlib.figure.Figure:

   data = data.t().detach().cpu().numpy()[1:] / 1000
   leadI = data[0]

   signals, info = neurokit2.ecg_process(leadI, sampling_rate = deepfakeecg.ECG_SAMPLING_RATE)
   neurokit2.ecg_plot(signals, info)

   # DIN A4 landscape: w=11.7, h=8.27
   w = 508/25.4   # mm to inch
   h = 122/25.4   # mm to inch
   matplotlib.pyplot.gcf().set_size_inches(w, h, forward=True)

   return matplotlib.pyplot.gcf()


# ###### Generic download ###################################################
def download(request:      gradio.Request,
             outputFormat: int) -> pathlib.Path | None:

   if outputFormat == deepfakeecg.OUTPUT_CSV:
      ecgResult = Sessions[request.session_hash].Results[Sessions[request.session_hash].Selected]
      ecgType   = Sessions[request.session_hash].Type
      fileName  = pathlib.Path(Sessions[request.session_hash].TempDirectory.name) / \
                     ('ECG-' + str(Sessions[request.session_hash].Selected + 1) + '.csv')
      deepfakeecg.dataToCSV(ecgResult, ecgType, fileName)

      log(f'Session "{request.session_hash}": Download CSV file {fileName}')
      return fileName

   elif ( (outputFormat == deepfakeecg.OUTPUT_PDF) or
          (outputFormat == deepfakeecg.OUTPUT_PDF_ANALYSIS) ):

      ecgResult = Sessions[request.session_hash].Results[Sessions[request.session_hash].Selected]
      ecgType   = Sessions[request.session_hash].Type
      fileName  = pathlib.Path(Sessions[request.session_hash].TempDirectory.name) / \
                     ('ECG-' + str(Sessions[request.session_hash].Selected + 1) + '.pdf')
      if ecgType == deepfakeecg.DATA_ECG12:
         outputLeads = [ 'I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4' , 'V5' , 'V6' ]
      else:
         outputLeads = [ 'I', 'II', 'V1', 'V2', 'V3', 'V4' , 'V5' , 'V6' ]

      deepfakeecg.dataToPDF(ecgResult, ecgType, outputLeads, fileName, outputFormat,
                            Sessions[request.session_hash].Selected + 1)

      log(f'Session "{request.session_hash}": Download PDF file {fileName}')
      return fileName

   return None


# ###### Download CSV #######################################################
def downloadCSV(request: gradio.Request) -> pathlib.Path | None:
   return download(request, deepfakeecg.OUTPUT_CSV)


# ###### Download PDF #######################################################
def downloadPDF(request: gradio.Request) -> pathlib.Path | None:
   return download(request, deepfakeecg.OUTPUT_PDF)


# ###### Download PDF #######################################################
def downloadPDFwithAnalysis(request: gradio.Request) -> pathlib.Path | None:
   return download(request, deepfakeecg.OUTPUT_PDF_ANALYSIS)


# ###### Analyze the selected ECG ###########################################
def analyze(event:   gradio.SelectData,
            request: gradio.Request) -> matplotlib.figure.Figure:

   Sessions[request.session_hash].Selected = event.index
   log(f'Session "{request.session_hash}": Analyze ECG #{Sessions[request.session_hash].Selected + 1}!')

   data = Sessions[request.session_hash].Results[Sessions[request.session_hash].Selected]
   if Sessions[request.session_hash].Analysis:
      matplotlib.pyplot.close(Sessions[request.session_hash].Analysis)
   Sessions[request.session_hash].Analysis = plotAnalysis(data)
   return Sessions[request.session_hash].Analysis


# ###### Print usage and exit ###############################################
def usage(exitCode : int = 0) -> str:
   sys.stdout.write('Usage: ' + sys.argv[0] + ' [-d|--device cpu|cuda] [-v|--version]\n')
   sys.exit(exitCode)



# ###### Main program #######################################################

# ====== Initialise =========================================================
runOnDevice: str = 'cuda' if torch.cuda.is_available() else 'cpu'
css = r"""
div {
   background-image: url("https://www.nntb.no/~dreibh/graphics/backgrounds/background-essen.png");
}

/* ###### General Settings ##############################################  */
html, body {
   height:           100%;
   margin:           0;
   padding:          0;
   font-family:      sans-serif;
   font-size:        small;
   background-color: #E3E3E3;   /* Simula background colour: #E3E3E3 */
   background-image: url("https://www.nntb.no/~dreibh/graphics/backgrounds/background-wiehl.png");
}


/* ###### Header ########################################################  */
div.program-header {
   background-image: none;
   background-color: #F15D22;   /* Simula header colour: #F15D22 */
   height:           7.5vh;
   display:          flex;
   justify-content:  space-between;
}

div.program-logo-left {
   width:            12.5vw;
   float:            left;
   display:          flex;
   padding:          0% 1%;
   align-items:      center;
   background:       white;
}

div.program-logo-right {
   width:            12.5vw;
   float:            right;
   display:          flex;
   padding:          0% 1%;
   align-items:      center;
   background:       white;
}

div.program-title {
   display:          flex;
   align-items:      center;
   padding:          0% 1%;
   background-image: none;
   background-color: #F15D22;   /* Simula header colour: #F15D22 */

   font-family:      "Open Sans", sans-serif;
   font-size:        4vh;
   font-weight:      bold;
}

img.program-logo-image {
   min-height:       4vh;
   max-height:       4vh;
   margin-left:      auto;
   margin-right:     auto;
}
"""


# ====== Check arguments ====================================================
try:
   options, args = getopt.gnu_getopt(
      sys.argv[1:],
      'd:v',
      [
         'device=',
         'version'
      ])
   for option, optarg in options:
      if option in ( '-d', '--device' ):
         runOnDevice = optarg
      elif option in ( '-v', '--version' ):
         sys.stdout.write('PyTorch version: ' + torch.__version__ + '\n')
         sys.stdout.write('CUDA version:    ' + torch.version.cuda + '\n')
         sys.stdout.write('CUDA available:  ' + ('yes' if torch.cuda.is_available() else 'no') + '\n')
         sys.stdout.write('Device:          ' + runOnDevice + '\n')
         sys.exit(1)
      else:
         sys.stderr.write('ERROR: Invalid option ' + option + '!\n')
         sys.exit(1)

except getopt.GetoptError as error:
   sys.stderr.write('ERROR: ' + str(error) + '\n')
   usage(1)
if len(args) > 0:
   usage(1)


# ====== Create GUI =========================================================
with gradio.Blocks(css               = css,
                   theme             = gradio.themes.Glass(secondary_hue=gradio.themes.colors.blue),
                   analytics_enabled = False,
                   fill_height       = True,
                   fill_width        = True) as gui:

   # ====== Session handling ================================================
   # Session initialization, to be called when page is loaded
   gui.load(initializeSession)
   # Session clean-up, to be called when page is closed/refreshed
   gui.unload(cleanUpSession)

   # ====== Header ==========================================================
   with gradio.Row(height = '10vh', min_height = '10vh', max_height = '10vh'):
      big_block = gradio.HTML("""
<div class="program-header">
   <div class="program-logo-left">
      <img class="program-logo-image" src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4NCjwhLS0gR2VuZXJhdG9yOiBBZG9iZSBJbGx1c3RyYXRvciAxNS4wLjAsIFNWRyBFeHBvcnQgUGx1Zy1JbiAuIFNWRyBWZXJzaW9uOiA2LjAwIEJ1aWxkIDApICAtLT4NCjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+DQo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxheWVyXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4Ig0KCSB3aWR0aD0iNDUyLjk3N3B4IiBoZWlnaHQ9IjEyNC43MjVweCIgdmlld0JveD0iMCAwIDQ1Mi45NzcgMTI0LjcyNSIgZW5hYmxlLWJhY2tncm91bmQ9Im5ldyAwIDAgNDUyLjk3NyAxMjQuNzI1Ig0KCSB4bWw6c3BhY2U9InByZXNlcnZlIj4NCjxnPg0KCTxnPg0KCQk8cGF0aCBmaWxsPSIjRjE1QTIyIiBkPSJNNDMzLjg1NCw5Mi42OTJjMCwxMS4yNi03LjQ0OCwxNi42MjktMTguMTg4LDE2LjYyOWMtOS43LDAtMTQuMjA0LTQuNTA0LTE0LjIwNC0xMS4yNTkNCgkJCWMwLTYuNTgyLDQuMTU3LTkuMDA3LDEzLjUxMi0xMC45MTNsNy42MjEtMS41NThjNS44ODktMS4yMTMsOC4zMTMtMS45MDYsMTEuMjU5LTIuOTQ1VjkyLjY5MnogTTQ1Mi43MzQsMTAzLjA4NlY2Mi4yMDgNCgkJCWMwLTE4LjE4OS0xMy44NTctMjQuMjUtMzIuNzM3LTI0LjI1Yy0xOS43NDYsMC0zNi4zNzUsOC4xNDEtMzYuMDI3LDI4LjQwOGwxOC41MzIsMS4yMTJjLTAuMTczLTkuNzAxLDUuNTQzLTE0LjM3NiwxNi42MjktMTQuMzc2DQoJCQljMTAuMjIsMCwxNC43MjQsMy40NjQsMTQuNzI0LDEwLjM5MnYxLjM4NmMwLDMuMTE4LTAuODY2LDMuNDY0LTUuMTk2LDQuMzNsLTE4LjE4OCwzLjExOGMtMTAuNzM5LDEuOTA2LTE5LjQsNS4xOTctMjQuNDI0LDExLjI1OQ0KCQkJYy0yLjk0MywzLjYzOS00LjY3Nyw4LjMxNC00LjY3NywxNC41NWMwLDE4LjAxNSwxMy4xNjUsMjUuOTgyLDI5LjEwMSwyNS45ODJjMTQuMTg4LDAsMTkuNzkzLTcuMDA4LDIzLjQxLTEwLjU3OXY4LjA2MmgxOC44NDQNCgkJCXYtMTguNTA2QzQ1Mi43MjcsMTAzLjE2LDQ1Mi43MzQsMTAzLjEyMSw0NTIuNzM0LDEwMy4wODYgTTU5Ljk3NCw2Mi43NzdjLTUuNzE3LTUuNTQzLTEzLjY4NC05LjM1My0yMy4yMTEtOS4zNTMNCgkJCWMtNy43OTUsMC0xMy41MTEsMi43NzEtMTMuNTExLDguMTQxYzAsNS44OSw1LjcxNiw3LjEwMiwxNy4zMjIsOS41MjZsOS44NzMsMi4yNTJjMTIuNDcxLDIuOTQ1LDIzLjczLDcuNjIyLDIzLjczLDIyLjY5MQ0KCQkJYzAsMTguODgtMTcuMTQ4LDI3Ljg4Ny0zNi4wMjgsMjcuODg3Yy0xOS43NDcsMC0zMy4wODQtOS42OTktMzguMTA3LTE2LjgwMWwxMy4zMzctMTEuNzc5DQoJCQljNC42NzcsNi4yMzYsMTIuNjQ1LDEyLjI5OCwyNS4yODksMTIuMjk4YzkuODczLDAsMTUuNTg5LTMuNDY0LDE1LjU4OS05LjM1M2MwLTUuNzE3LTQuNjc2LTYuNzU1LTEyLjY0NC04LjQ4OGwtMTIuMjk4LTIuNTk4DQoJCQljLTE0LjM3Ny0zLjExOC0yNS42MzYtOS4xODEtMjUuNjM2LTI0LjI1YzAtMTguMDE1LDE3LjQ5NS0yNS42MzYsMzMuMjU3LTI1LjYzNmMxNi40NTUsMCwyOS4xLDcuMjc1LDM0LjQ3LDEzLjE2NEw1OS45NzQsNjIuNzc3eg0KCQkJIE0xODcsNDUuMjg0Yy00Ljg1LTUuMzctMTEuOTUyLTcuOTY5LTIwLjQzOS03Ljk2OWMtMTIuOTkxLDAtMjEuODI1LDUuNzE3LTI1LjgwOSwxMi42NDVWMzkuMzkzaC0xOS43NDZ2ODIuNDUxaDE5Ljc0NlY3Ny42NzUNCgkJCWMwLTEzLjMzOCw1LjcxNi0yMi44NjYsMTcuODQxLTIyLjg2NmMxNC44OTYsMCwxNS45MzYsMTAuMzk0LDE1LjkzNiwyNS4yOTF2NDEuNzQ0aDE5Ljc0NlY3Ny42NzUNCgkJCWMwLTEzLjMzOCw1LjU0My0yMi44NjYsMTcuODQxLTIyLjg2NmMxNC44OTcsMCwxNS45MzgsMTAuMzk0LDE1LjkzOCwyNS4yOTF2NDEuNzQ0SDI0Ny44Vjc1LjA3Ng0KCQkJYzAtMTguMTg4LTIuMDc4LTIzLjkwNC03LjI3NC0yOS43OTJjLTQuMzMxLTQuODUxLTExLjI1OS03Ljk2OS0yMS44MjctNy45NjljLTE0Ljg5NiwwLTIzLjIxMSw2Ljc1NS0yOC4yMzQsMTIuODE4DQoJCQlDMTg5LjQyNSw0OC40LDE4OC4zODYsNDYuODQyLDE4Nyw0NS4yODQgTTI2MS44NTMsMzkuMzkzaDE5Ljc0NnY0My44MjRjMCw1Ljg5LDAuMTc0LDExLjYwNiwyLjA3OCwxNS45MzYNCgkJCWMxLjkwNiw0LjMzLDUuNzE3LDcuMjc1LDEyLjY0Niw3LjI3NWMxMi4yOTgsMCwxNy42NjgtOS41MjcsMTcuNjY4LTIyLjg2NHYtNDQuMTdoMTkuNzQ2djgyLjQ1MUgzMTMuOTl2LTEwLjU2Ng0KCQkJYy0zLjk4NCw2LjkyOS0xMi44MTcsMTIuNjQ1LTI0LjU5NywxMi42NDVjLTguMzE0LDAtMTUuNDE2LTIuNTk4LTIwLjI2Ni03Ljk2N2MtNS4xOTctNS44OTEtNy4yNzUtMTEuNjA2LTcuMjc1LTI5Ljc5M1YzOS4zOTN6DQoJCQkgTTEwNS4zNTUsMzkuMzgySDg1LjU5MXY4Mi40NjFoMTkuNzY0VjM5LjM4MnogTTM2OS4wNjksMTIxLjg0NGgtMTkuNzQ2VjQuMDU4aDE5Ljc0NlYxMjEuODQ0eiBNODIuMTIyLDEzLjk4Nw0KCQkJYzAsNy4zOTQsNi4wMjQsMTMuNjA5LDEzLjM1MSwxMy42MDljNy40OSwwLDEzLjM1Mi02LjIxNiwxMy4zNTItMTMuNjA5YzAtNy43MjktNS44NjItMTMuOTQ0LTEzLjM1Mi0xMy45NDQNCgkJCUM4OC4xNDYsMC4wNDIsODIuMTIyLDYuMjU4LDgyLjEyMiwxMy45ODciLz4NCgk8L2c+DQo8L2c+DQo8L3N2Zz4NCg==" alt="SimulaMet" height="32" />
   </div>
   <div class="program-title" id="title"><a href="https://ihi-search.eu/">SEARCH</a>&nbsp;DeepFake ECG Generator v""" + version.DEEPFAKEECGGENPLUS_VERSION +  """</div>
   <div class="program-logo-right">
      <img class="program-logo-image" src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4NCjwhLS0gR2VuZXJhdG9yOiBBZG9iZSBJbGx1c3RyYXRvciAxNS4wLjAsIFNWRyBFeHBvcnQgUGx1Zy1JbiAuIFNWRyBWZXJzaW9uOiA2LjAwIEJ1aWxkIDApICAtLT4NCjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+DQo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxheWVyXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4Ig0KCSB3aWR0aD0iMTAzNy44OTVweCIgaGVpZ2h0PSIxNzQuMjVweCIgdmlld0JveD0iMCAwIDEwMzcuODk1IDE3NC4yNSIgZW5hYmxlLWJhY2tncm91bmQ9Im5ldyAwIDAgMTAzNy44OTUgMTc0LjI1Ig0KCSB4bWw6c3BhY2U9InByZXNlcnZlIj4NCjxnPg0KCTxwYXRoIGZpbGw9IiNGMTVBMjIiIGQ9Ik0xMTUuMTAzLDIwLjkzN2MwLDEwLjIyNiw4LjM0MywxOC44NCwxOC40ODEsMTguODRjMTAuMzcxLDAsMTguNDcyLTguNjE0LDE4LjQ3Mi0xOC44NA0KCQljMC0xMC42OTEtOC4xMDEtMTkuMjk2LTE4LjQ3Mi0xOS4yOTZDMTIzLjQ0NSwxLjY0MSwxMTUuMTAzLDEwLjI0NSwxMTUuMTAzLDIwLjkzNyBNNTEyLjI1NiwxNzAuMjIyaC0yNy4zMzdWNy4xOTloMjcuMzM3VjE3MC4yMjINCgkJeiBNMTQ3LjI1Miw1Ni4wNzVoLTI3LjM1OHYxMTQuMTI3aDI3LjM1OFY1Ni4wNzV6IE0zNjMuODY2LDU2LjA5NWgyNy4zMzh2NjAuNjcyYzAsOC4xMywwLjIzMywxNi4wNDYsMi44NzEsMjIuMDQyDQoJCWMyLjYzOSw1Ljk3Niw3LjkxNiwxMC4wNjgsMTcuNSwxMC4wNjhjMTcuMDE3LDAsMjQuNDQ4LTEzLjE5MiwyNC40NDgtMzEuNjQ1VjU2LjA5NWgyNy4zMzl2MTE0LjEyN2gtMjcuMzM5di0xNC42NDkNCgkJYy01LjUwOSw5LjYwNS0xNy43NTIsMTcuNTIxLTM0LjAzMiwxNy41MjFjLTExLjUwNiwwLTIxLjM0My0zLjYwOC0yOC4wNTctMTEuMDQxYy03LjE5Ny04LjE0Ny0xMC4wNjgtMTYuMDQ1LTEwLjA2OC00MS4yM1Y1Ni4wOTV6DQoJCSBNMjU5LjU0NSw2NC4yNDNjLTYuNzEzLTcuNDMxLTE2LjU1LTExLjAyMS0yOC4yODktMTEuMDIxYy0xNy45ODYsMC0zMC4xOTEsNy44OTYtMzUuNzIsMTcuNTAydi0xNC42M2gtMjcuMzE5djExNC4xMjdoMjcuMzE5DQoJCXYtNjEuMTM5YzAtMTguNDcyLDcuOTE2LTMxLjY0NiwyNC42OTktMzEuNjQ2YzIwLjYyNSwwLDIyLjA2MSwxNC4zNzcsMjIuMDYxLDM1LjAwMnY1Ny43ODJoMjcuMzM4di02MS4xMzkNCgkJYzAtMTguNDcyLDcuNjY1LTMxLjY0NiwyNC42ODEtMzEuNjQ2YzIwLjYwNSwwLDIyLjA2MSwxNC4zNzcsMjIuMDYxLDM1LjAwMnY1Ny43ODJoMjcuMzE5di02NC43MjkNCgkJYzAtMjUuMTY1LTIuODUzLTMzLjEwMS0xMC4wNTEtNDEuMjVjLTUuOTk2LTYuNzEzLTE1LjYtMTEuMDIxLTMwLjIyOS0xMS4wMjFjLTIwLjYwNiwwLTMyLjExMiw5LjM1Mi0zOS4wNzcsMTcuNzMzDQoJCUMyNjIuOTAyLDY4LjU1MiwyNjEuNDg1LDY2LjQxNiwyNTkuNTQ1LDY0LjI0MyBNODQuNDQ2LDg4LjQ1OEM3Ni41NCw4MC43OTQsNjUuNTEsNzUuNTE3LDUyLjMxNSw3NS41MTcNCgkJYy0xMC43ODgsMC0xOC42OTQsMy44MjItMTguNjk0LDExLjI1NGMwLDguMTY4LDcuOTA2LDkuODM3LDIzLjk4MSwxMy4xOTNsMTMuNjUsMy4xMjRjMTcuMjY5LDQuMDc0LDMyLjg2OCwxMC41NTUsMzIuODY4LDMxLjQxMg0KCQljMCwyNi4xMzYtMjMuNzQ5LDM4LjU5My00OS44NzQsMzguNTkzYy0yNy4zMzksMC00NS44LTEzLjQyNy01Mi43NDYtMjMuMjQ0bDE4LjQ1MS0xNi4zMTgNCgkJYzYuNDgxLDguNjM2LDE3LjUwMiwxNy4wMTcsMzUuMDEzLDE3LjAxN2MxMy42NjksMCwyMS41NzUtNC43NzMsMjEuNTc1LTEyLjkyNGMwLTcuOTM1LTYuNDgtOS4zNy0xNy41MDEtMTEuNzU3bC0xNy4wMzUtMy42MDgNCgkJQzIyLjExNiwxMTcuOTUsNi41NDQsMTA5LjU2OCw2LjU0NCw4OC43MWMwLTI0LjkzMywyNC4xOTUtMzUuNDg3LDQ2LjAxNC0zNS40ODdjMjIuNzg4LDAsNDAuMjcsMTAuMDY5LDQ3LjcyMSwxOC4yMkw4NC40NDYsODguNDU4DQoJCXogTTYyOC4wNywxNDQuMjZWODcuNjYyYzAtMjUuMTY1LTE5LjIwOS0zMy41NDctNDUuMzI0LTMzLjU0N2MtMjcuMzM5LDAtNTAuMzUsMTEuMjUzLTQ5Ljg2NSwzOS4zMTFsMjUuNjUsMS42NjgNCgkJYy0wLjI1My0xMy40MjcsNy42NjUtMTkuODg4LDIzLjAxMi0xOS44ODhjMTQuMTYzLDAsMjAuMzczLDQuNzkyLDIwLjM3MywxNC4zOTZ2MS45MDJjMCw0LjMwNy0xLjE4Myw0Ljc5Mi03LjE5OCw1Ljk5NA0KCQlsLTI1LjE2Niw0LjMyN2MtMTQuODYxLDIuNjQtMjYuODM0LDcuMTgtMzMuOCwxNS41NjJjLTQuMDczLDUuMDQ1LTYuNDc5LDExLjUyNC02LjQ3OSwyMC4xNTkNCgkJYzAsMjQuOTMzLDE4LjIxOSwzNS45NTMsNDAuMjc4LDM1Ljk1M2MxOS42NTUsMCwyNy4zOTYtOS43MDEsMzIuNDAzLTE0LjY0OHYxMS4xNzZoMjYuMDk4di0yNS42MzENCgkJQzYyOC4wNTMsMTQ0LjM1Nyw2MjguMDcsMTQ0LjMsNjI4LjA3LDE0NC4yNiBNNjAxLjkxNiwxMjkuODYzYzAsMTUuNTgtMTAuMjg0LDIzLjAxMi0yNS4xNjYsMjMuMDEyDQoJCWMtMTMuNDI3LDAtMTkuNjUzLTYuMjI5LTE5LjY1My0xNS41ODFjMC05LjExOCw1Ljc2Mi0xMi40NzYsMTguNzA0LTE1LjA5NWwxMC41MzMtMi4xNTRjOC4xNS0xLjY2OCwxMS41MDctMi42NTcsMTUuNTgyLTQuMDczDQoJCVYxMjkuODYzeiIvPg0KCTxwYXRoIGZpbGw9IiMyNDEzNjMiIGQ9Ik0xMDM2LjUyNSwxNDkuMzgxYy0yLjg4NCwwLjQ4LTcuMjA5LDAuOTYxLTEyLjAxNiwwLjk2MWMtNC44MDYsMC05LjM3MS0wLjcyMS0xMS41MzQtMy4xMjQNCgkJYy0yLjE2My0yLjE2Mi0zLjEyNC01LjI4Ni0zLjEyNC0xMS43NzRWNzcuMDUyaDI2LjY3NFY1NS42NjZoLTI2LjY3NFY5Ljc3bC0yNy4zOTMsNi4wMDd2MzkuODloLTI0LjUxdjIxLjM4NmgyNC41MXY2Mi45NTgNCgkJYzAsMTEuMDUzLDAuMjQsMTcuNTQxLDQuODA1LDIzLjA2N2M2LjI0OCw3LjY4OSwxNy43ODIsOS44NTMsMjguODM2LDkuODUzYzguMTcsMCwxNS4xMzktMC43MjEsMjAuNDI2LTEuOTIyVjE0OS4zODF6DQoJCSBNODk5LjMxNiw3NC44OWMxNS4zNzksMCwyNi4xOTEsMTAuMDkyLDI2LjkxMiwyNC45OWgtNTUuNzQ5Qzg3MS4yMDEsODYuMTgzLDg4My42OTcsNzQuODksODk5LjMxNiw3NC44OSBNOTM0Ljg3OSwxMjkuNjc3DQoJCWMtMy42MDQsNS43NjctMTIuNzM1LDE5LjcwNC0zMy40MDEsMTkuNzA0Yy0yMC4xODQsMC0yOS43OTYtMTMuNDU2LTMxLjcxOS0yOS4wNzZoODMuMzgzYzAtMS45MjIsMC4yNC01LjA0NiwwLjI0LTYuOTY4DQoJCWMwLTM1LjA4My0xNy41NDEtNjAuMzE0LTU0LjA2NS02MC4zMTRjLTMxLjk2MSwwLTU2LjIyOSwyMy41NDktNTYuMjI5LDU4Ljg3MmMwLDM3LjAwNSwyMy4wNjgsNjEuMDM1LDU2Ljk1LDYxLjAzNQ0KCQljMzMuNCwwLDQ4Ljc3OS0yMC45MDYsNTMuODI2LTI4LjgzNUw5MzQuODc5LDEyOS42Nzd6IE03NDAuOTYyLDYzLjgzNmMtNi43MjgtNy40NDktMTYuNTgtMTEuMDU1LTI4LjM1NS0xMS4wNTUNCgkJYy0xOC4wMjEsMC0zMC4yNzYsNy45MzEtMzUuODA0LDE3LjU0MlY1NS42NjZINjQ5LjQxdjExNC4zOGgyNy4zOTN2LTYxLjI3NWMwLTE4LjUwMyw3LjkzLTMxLjcxOSwyNC43NTEtMzEuNzE5DQoJCWMyMC42NjUsMCwyMi4xMDcsMTQuNDE4LDIyLjEwNywzNS4wODN2NTcuOTExaDI3LjM5M3YtNjEuMjc1YzAtMTguNTAzLDcuNjktMzEuNzE5LDI0Ljc1MS0zMS43MTkNCgkJYzIwLjY2NSwwLDIyLjEwNywxNC40MTgsMjIuMTA3LDM1LjA4M3Y1Ny45MTFoMjcuMzk0di02NC44NzljMC0yNS4yMzEtMi44ODQtMzMuMTYxLTEwLjA5My00MS4zMzENCgkJYy02LjAwNy02LjcyOS0xNS42MTktMTEuMDU1LTMwLjI3Ny0xMS4wNTVjLTIwLjY2NSwwLTMyLjE5OSw5LjM3Mi0zOS4xNjgsMTcuNzgyQzc0NC4zMjYsNjguMTYsNzQyLjg4NCw2NS45OTksNzQwLjk2Miw2My44MzYiLz4NCjwvZz4NCjwvc3ZnPg0K" alt="NorNet" height="64" />
   </div>
</div>
""")

   # gradio.Markdown('## Settings')

   with gradio.Row(height = '10vh', min_height = '10vh', max_height = '10vh'):
      sliderNumberOfECGs     = gradio.Slider(1, 100, label="Number of ECGs", step = 1, value = 4, interactive = True)
      # sliderLengthInSeconds = gradio.Slider(5, 60, label="Length (s)", step = 5, value = 10, interactive = True)
      dropdownType           = gradio.Dropdown( [ 'ECG-12', 'ECG-8' ], label = 'ECG Type', interactive = True)
      dropdownGeneratorModel = gradio.Dropdown( [ 'Default' ], label = 'Generator Model', interactive = True)
      with gradio.Column():
         buttonGenerate = gradio.Button("Generate ECGs!")
         # buttonAnalyze  = gradio.Button("Analyze this ECG!")
         with gradio.Row():
            buttonCSV = gradio.DownloadButton("Download CSV")
            buttonCSV_hidden = gradio.DownloadButton(visible=False, elem_id="download_csv_hidden")
            buttonPDF = gradio.DownloadButton("Download ECG PDF")
            buttonPDF_hidden = gradio.DownloadButton(visible=False, elem_id="download_pdf_hidden")
            buttonPDFwAnalysis = gradio.DownloadButton("Download ECG+Analysis PDF")
            buttonPDFwAnalysis_hidden = gradio.DownloadButton(visible=False, elem_id="download_pdfwanalysis_hidden")

   # gradio.Markdown('## Output')

   with gradio.Row(): # height = '24vh', min_height = '24vh', max_height = '24vh'):
      outputGallery = gradio.Gallery(label         = 'Generated ECGs',
                                     columns       = 8,
                                     # rows          = 1,
                                     height        = 'auto',
                                     object_fit    = 'contain',
                                     show_label    = True,
                                     allow_preview = True,
                                     preview       = False
                                    )

   with gradio.Row(): # height = '24vh', min_height = '24vh', max_height = '24vh'):
      analysisOutput = gradio.Plot(label = 'Analysis')

   # ====== Add click event handling for "Generate" button ==================
   buttonGenerate.click(predict,
                        inputs  = [ sliderNumberOfECGs,
                                    # sliderLengthInSeconds,
                                    dropdownType,
                                    dropdownGeneratorModel ],
                        outputs = [ outputGallery, analysisOutput ]
                     )

   # ====== Add click event handling for "Analyze" button ===================
   outputGallery.select(analyze,
                        inputs  = [ ],
                        outputs = [ analysisOutput ]
                       )

   # ====== Add click event handling for download buttons ===================
   # Using hidden button and JavaScript, to generate download file on-the-fly:
   # https://github.com/gradio-app/gradio/issues/9230#issuecomment-2323771634
   buttonCSV.click(fn      = downloadCSV,
                   inputs  = None,
                   outputs = [ buttonCSV_hidden ]).then(
                      fn = None, inputs = None, outputs = None,
                      js = "() => document.querySelector('#download_csv_hidden').click()")
   buttonPDF.click(fn      = downloadPDF,
                   inputs  = None,
                   outputs = [ buttonPDF_hidden ]).then(
                      fn = None, inputs = None, outputs = None,
                      js = "() => document.querySelector('#download_pdf_hidden').click()")
   buttonPDFwAnalysis.click(fn      = downloadPDFwithAnalysis,
                            inputs  = None,
                            outputs = [ buttonPDFwAnalysis_hidden ]).then(
                               fn = None, inputs = None, outputs = None,
                               js = "() => document.querySelector('#download_pdfwanalysis_hidden').click()")

   # ====== Run on startup ==================================================
   gui.load(predict,
            inputs  = [ sliderNumberOfECGs,
                        # sliderLengthInSeconds,
                        dropdownType,
                        dropdownGeneratorModel ],
            outputs = [ outputGallery, analysisOutput ]
           )

# ====== Run the GUI ========================================================
if __name__ == "__main__":

   # ------ Prepare temporary directory -------------------------------------
   TempDirectory = tempfile.TemporaryDirectory(prefix = 'DeepFakeECGPlus-')
   log(f'Prepared temporary directory {TempDirectory.name}')

   # ------ Run the GUI, with downloads from temporary directory allowed ----
   gui.launch(allowed_paths = [ TempDirectory.name ], debug = True)

   # ------ Clean up --------------------------------------------------------
   log(f'Cleaning up temporary directory {TempDirectory.name}')
   TempDirectory.cleanup()
   log('Done!')
