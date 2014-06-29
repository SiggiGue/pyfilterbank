.. PyFilterbank documentation master file, created by
   sphinx-quickstart on Sun Jun  1 13:01:31 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: filtbank_logo.png
   :width: 591px
   :height: 157px
   :scale: 50%
   :target: http://github.com/SiggiGue/pyfilterbank

|
|
Welcome to PyFilterbank's documentation!
========================================


The package :mod:`pyfilterbank` provides tools for the acousticians and audiologists working with python.

A fractional octave filter bank is provided in the module :mod:`octbank`. You can use it to split your signals into many bands of constant relative fractional octave band width. The output signals stay in the same domain as the input signal but are band passed groups of it. The filtering routines are placed in :mod:`sosfiltering` and the filter design functionality is implemented in :mod:`butterworth`.

Spectral weigthing for level measurements can be done with the tools in :mod:`splweighting`.
For fft-based and more physiological motivated filtering there is the module :mod:`melbank` with some tools for transforming linear spectra to mel-spectra.

A gammatone filter bank and stft is planned but no implemented yet. If there is time and some other persons are intersted in contributing, there are many functionalites that can be added and maintained.

Have Fun!

GitHub Repo
-----------
http://github.com/SiggiGue/pyfilterbank

Content
-------
.. toctree::
   octbank
   melbank
   gammatone
   splweighting
   sosfiltering
   glossary


License
_______
pyfilterbank is licensed unter the BSD 4-clause license:

Copyright (c) 2014, Siegfried Gündert
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software
   must display the following acknowledgement:
   This product includes software developed by the <organization>.
4. Neither the name of the <organization> nor the
   names of its contributors may be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY <COPYRIGHT HOLDER> ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
