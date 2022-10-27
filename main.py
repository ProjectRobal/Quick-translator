import pyaudio
import webrtcvad
import stt
from googletrans import Translator
from googletrans.models import Translated
import threading
import queue
from scipy.io.wavfile import write
import numpy
import ctypes
import os
from tts_interface import TTSInterface

# config variables
from config import *

class AudioSamp:
    def __init__(self,buffer:numpy.array,in_lan:str,out_lan:str):
        self._buffer = buffer
        self._input_lan = in_lan
        self._output_lan = out_lan

    @property
    def input_lang(self):
        return self._input_lan

    @property
    def output_lang(self):
        return self._output_lan

    @property
    def buffer(self):
        return self._buffer

class TranslatorInput:
    def __init__(self,input_text:str,input_lang:str,output_lang:str,audio_samp:AudioSamp):
        self._input_lang = input_lang
        self._output_lang = output_lang
        self._input_text = input_text
        self._audio_samp=audio_samp

    def audio_samp(self):
        return self._audio_samp

    @property
    def input_lang(self):
        return self._input_lang

    @property
    def output_lang(self):
        return self._output_lang

    @property
    def text(self):
        return self._input_text

class TTSInput:
    def __init__(self,input_text:str,output_lang:str,audio_samp:AudioSamp):
        self._output_lang = output_lang
        self._input_text = input_text
        self._audio_samp= audio_samp

    def audio_samp(self):
        return self._audio_samp

    @property
    def output_lang(self):
        return self._output_lang

    @property
    def text(self):
        return self._input_text


class Speaker:

    def __init__(self,chunk,format=pyaudio.paInt16,channels=1,rate=16000,id=0):

        self.audio=pyaudio.PyAudio()
        self.open_stream(format,channels,rate,chunk,id)

    def open_stream(self,format,channels,rate,chunk,id=0):

        self.close_stream()

        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.id = id

        self.stream=self.audio.open(format=format, channels=channels,rate=rate,frames_per_buffer=chunk,output=True,output_device_index=id)
        self.chunk=chunk

    def put(self,sample:numpy.array):
        self.stream.write(sample,len(sample),exception_on_overflow = False)

    '''Close current stream'''
    def close_stream(self):
        if hasattr(self,'stream'):
            self.stream.stop_stream()
            self.stream.close()

    '''Close speaker completely'''
    def close(self):
        self.close_stream()
        self.audio.terminate()
        



class Microphone:
    def __init__(self,chunk,format=pyaudio.paInt16,channels=1,rate=16000,id=0):

        '''init input audio device'''
        self.audio=pyaudio.PyAudio()
        self.open_stream(format,channels,rate,chunk,id)

        '''vad initialization'''

        self.vad=webrtcvad.Vad()
        self.vad.set_mode(2)

    '''Change vad aggresive level'''
    def vad_mode(self,mode):
        self.vad.set_mode(mode)

    '''Reopen a stream with different settings'''
    def open_stream(self,format,channels,rate,chunk,id=0):

        self.close_stream()

        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.id = id

        self.stream=self.audio.open(format=format, channels=channels,rate=rate,frames_per_buffer=chunk,input=True,input_device_index=id)
        self.chunk=chunk

    '''Get a chunk then check if it has a voice , if it is true it return a array with chunk else it returns None '''
    def get(self):
        frame=self.stream.read(self.chunk,exception_on_overflow = False)

        if self.vad.is_speech(frame,self.rate):
            return numpy.frombuffer(frame,dtype=numpy.int16)

        return None

    '''Close current stream'''
    def close_stream(self):
        if hasattr(self,'stream'):
            self.stream.stop_stream()
            self.stream.close()

    '''Close microphone completely'''
    def close(self):
        self.close_stream()
        self.audio.terminate()

    



class QuickTrans:
    def __init__(self,mic:Microphone,speaker:Speaker,langs):
        self._langs = langs
        self._input_lang=self.check_input_langs()
        self._output_lang=self.check_output_langs()
        self._mic = mic

        self.stt_queue = queue.Queue(20) #a queue for stt thread
        self.stt_th=None

        self.tts_queue = queue.Queue(20) #a queue for tts thread
        self.tts_th=None

        self.trans_queue = queue.Queue(20) #a queue for translator thread
        self.trans_th=None

        self.clear_buffer()
        self.speech_begin=False
        self.recording=False

        self._speaker=speaker

    def check_input_langs(self):
        entries=os.listdir(STT_DIRECTORY+"/")

        langs=[]

        for entry in entries:
            if os.path.isdir(STT_DIRECTORY+"/"+entry):
                langs.append(entry)

        return langs

    def check_output_langs(self):
        entries=os.listdir(TTS_DIRECTORY+"/")

        langs=[]

        for entry in entries:
            if os.path.isdir(TTS_DIRECTORY+"/"+entry):
                langs.append(entry)

        return langs

    
    def supported_languages(self):
        '''a array with language strings like 'pl','en' itd. '''
        return self._langs

    def input_language(self,input:str or None =None):
        '''set a input language'''
        if input is None:
            return self._input_lang
        
        if input in self._langs:
            self._input_lang=input
        else:
            raise ValueError("No such language: "+input)

    
    def output_language(self,output:str or None =None):
        '''set a output language'''
        if output is None:
            return self._input_lang
        
        if output in self._langs:
            self._output_lang=output
        else:
            raise ValueError("No such language: "+output)

    
    def clear_buffer(self):
        '''clear buffer with samples'''
        self.buffer=numpy.empty(0,dtype=numpy.int16)

    def concat_buffer(self,sample):
        '''append samples to buffer'''
        self.buffer=numpy.concatenate((self.buffer,sample))

    def start_threads(self):
        '''start stt and tts threads'''
        self.stt_th=threading.Thread(target=self.stt_thread)

        self.stt_th.start()

        self.trans_th=threading.Thread(target=self.trans_thread)

        self.trans_th.start()

        self.tts_th=threading.Thread(target=self.tts_thread)

        self.tts_th.start()

    def close_threads(self):
        '''close stt and tts threads'''
        ctypes.pythonapi.PyThreadState_SetAsyncExc(self.stt_th.ident,ctypes.py_object(SystemExit))

        ctypes.pythonapi.PyThreadState_SetAsyncExc(self.tts_th.ident,ctypes.py_object(SystemExit))

        ctypes.pythonapi.PyThreadState_SetAsyncExc(self.trans_th.ident,ctypes.py_object(SystemExit))

    def speech_to_text(self,audio,input_lang:str)->str:
        '''perform stt , audio - is a input audio data (numpy.array int16) , input_lang - a source language'''
        model=stt.Model('STT/'+input_lang+'/model.tflite')

        model.disableExternalScorer()

        print("Processing...")
        output=model.stt(audio)

        print("Finished")
        return output

    def translate(self,text:str,input_lang:str,output_lang:str)->Translated:
        '''perform text translation from input language (input_lang) to output language (output_lang)'''
        translator=Translator()
        return translator.translate(text,dest=output_lang,src=input_lang)

    def stt_thread(self):
        '''stt thread function'''
        while True:
            '''get audio sample from queue'''
            item=self.stt_queue.get()
            '''check if audio sample was accquired'''
            if type(item) == AudioSamp:
                '''perform speech to text operation'''
                output=self.speech_to_text(item.buffer,item.input_lang)

                print(output)
                '''check if stt generated valid output'''
                if type(output)==str and len(output)>0:
                    '''then translate it'''                    
                    self.put_to_translator(TranslatorInput(output,item.input_lang,item.output_lang,item))

            self.stt_queue.task_done()

    def put_to_stt(self,smp:AudioSamp):
        '''put audio sample to stt thread'''
        self.stt_queue.put(smp)

    def trans_thread(self):
        '''google translate thread'''

        while True:
            item:TranslatorInput=self.trans_queue.get()

            if len(item.text)>0:
                    '''then translate it'''
                    translated=self.translate(item.text,item.input_lang,item.output_lang)
                    print(translated.text)

                    self.put_to_tts(TTSInput(translated,item.input_lang,item.output_lang,item.audio_samp()))
            
            self.trans_queue.task_done()


    def put_to_translator(self,item:TranslatorInput):
        self.trans_queue.put(item)

    def tts_thread(self):
        '''tts thread function'''
        while True:
            item:TTSInput=self.tts_queue.get()

            text=item.text

            output_lang=item.output_lang

            samp=item.audio_samp().buffer

            # cut it off
            if len(samp)> 16000*5:
                samp=samp[0:16000*5]

            try:
                
                interface=TTSInterface(TTS_DIRECTORY+"/"+output_lang+"/")

                (wav,alignment)=interface.do_tts(text,samp)

                self._speaker.put(wav)

            except Exception as e:
                print(e)

            self.tts_queue.task_done()

    def put_to_tts(self,item:TTSInput):
        self.tts_queue.put(item)

    def mic(self):
        '''get microphone object'''
        return self._mic

    def speaker(self):
        '''get speaker object'''
        return self._speaker

    def startRecording(self):
        self.speech_begin=True

    def main_task(self):

        '''wait for stt queue to be empty'''
        if self.stt_queue.full():
            return
        
        if self.recording:
            '''get voice samples'''
            sample=self.mic().get()

            if sample is not None:
                self.concat_buffer(sample)
                print("voice!!")
                self.speech_begin=True
        elif self.speech_begin:
            self.put_to_stt(AudioSamp(self.buffer,self.input_language(),self.output_language()))
            self.speech_begin=False
            self.clear_buffer()

                




FORMAT= pyaudio.paInt16
CHANNELS = 1
RATE=16000


FRAME_DURATION=10 # 10 ms

CHUNK=int(FRAME_DURATION*RATE/1000)

'''We want to record about 10 ms of audio for each check'''


mic=Microphone(CHUNK,FORMAT,CHANNELS,RATE)

print("Microphone ready!!")

quick=QuickTrans(mic)

print("Quick translator setuped!!")

quick.output_language('de')

quick.input_language('en')

quick.start_threads()

print("STT and TTS threads have been started!")

try:

    while True:

        quick.main_task()


except Exception as e:
    print(e)
    quick.close_threads()
    mic.close()


