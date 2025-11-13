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

DEEPFAKEECGGENPLUS_VERSION_MAJOR = 0
DEEPFAKEECGGENPLUS_VERSION_MINOR = 6
DEEPFAKEECGGENPLUS_VERSION_PATCH = "0~rc2"
DEEPFAKEECGGENPLUS_VERSION       = str(DEEPFAKEECGGENPLUS_VERSION_MAJOR) + '.' + \
                                   str(DEEPFAKEECGGENPLUS_VERSION_MINOR) + '.' + \
                                   str(DEEPFAKEECGGENPLUS_VERSION_PATCH)
DEEPFAKEECGGENPLUS_PACKAGE       = 'deepfake-ecg-generator-plus-' + DEEPFAKEECGGENPLUS_VERSION
