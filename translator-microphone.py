import argparse
import sys
import signal
from datetime import datetime
import numpy as np
from audioop import rms
from pyaudio import PyAudio, paInt16
import whisper
from whisper.audio import SAMPLE_RATE
# from pygoogletranslation import Translator


class RingBuffer:
    def __init__(self, size):
        self.size = size
        self.data = []
        self.full = False
        self.cur = 0

    def append(self, x):
        if self.size <= 0:
            return
        if self.full:
            self.data[self.cur] = x
            self.cur = (self.cur + 1) % self.size
        else:
            self.data.append(x)
            if len(self.data) == self.size:
                self.full = True

    def get_all(self):
        """ Get all elements in chronological order from oldest to newest. """
        all_data = []
        for i in range(len(self.data)):
            idx = (i + self.cur) % self.size
            all_data.append(self.data[idx])
        return all_data

    def has_repetition(self):
        prev = None
        for elem in self.data:
            if elem == prev:
                return True
            prev = elem
        return False

    def clear(self):
        self.data = []
        self.full = False
        self.cur = 0


def open_mic_stream(device_index, preferred_quality):
    pa = PyAudio()
    input_stream = pa.open(format=paInt16, channels=1, rate=SAMPLE_RATE, input=True,
                           frames_per_buffer=SAMPLE_RATE // 2, input_device_index=device_index)

    process = None  # This is the FFmpeg process, which is not used here

    return input_stream, process


def main(device_index, model="base", language=None, interval=5, history_buffer_size=0, preferred_quality="audio_only",
         use_vad=True, faster_whisper_args=None, **decode_options):

    n_bytes = interval * SAMPLE_RATE * 2  # Factor 2 comes from reading the int16 stream as bytes
    audio_buffer = RingBuffer((history_buffer_size // interval) + 1)
    previous_text = RingBuffer(history_buffer_size // interval)

    print("Loading speech recognition model...")
    if faster_whisper_args:
        from faster_whisper import WhisperModel
        model = WhisperModel(faster_whisper_args["model_path"],
                             device=faster_whisper_args["device"],
                             compute_type=faster_whisper_args["compute_type"])
    else:
        model = whisper.load_model(model)
    print("done")

    if use_vad:
        from vad import VAD
        vad = VAD()

    translate_to_Chinese=decode_options['translate_to_Chinese']
    decode_options.pop("translate_to_Chinese")
    print(translate_to_Chinese)
    if translate_to_Chinese==True:
        print("Loading translation model...")
        from transformers import MarianTokenizer, MarianMTModel
        # 指定模型名称
        model_name = "Helsinki-NLP/opus-mt-en-zh"
        # 加载模型和分词器
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        translate_model = MarianMTModel.from_pretrained(model_name)
        print("done")


    print("Opening microphone stream...")
    mic_stream, _ = open_mic_stream(device_index, preferred_quality)

    def handler(signum, frame):
        mic_stream.stop_stream()
        mic_stream.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

    # translator = Translator()

    print("begin to recognize")
    try:
        while True:
            in_bytes = mic_stream.read(n_bytes)
            audio = np.frombuffer(in_bytes, np.int16).flatten().astype(np.float32) / 32768.0
            if use_vad and vad.no_speech(audio):
                print(f'{datetime.now().strftime("%H:%M:%S")}')
                continue
            audio_buffer.append(audio)

            # Decode the audio
            clear_buffers = False
            if faster_whisper_args:
                segments, info = model.transcribe(audio,
                                                  language=language,
                                                  **decode_options)

                decoded_language = "" if language else "(" + info.language + ")"
                decoded_text = ""
                previous_segment = ""
                for segment in segments:
                    if segment.text != previous_segment:
                        decoded_text += segment.text
                        previous_segment = segment.text

                new_prefix = decoded_text

            else:
                result = model.transcribe(np.concatenate(audio_buffer.get_all()),
                                          prefix="".join(previous_text.get_all()),
                                          language=language,
                                          without_timestamps=True,
                                          **decode_options)

                decoded_language = "" if language else "(" + result.get("language") + ")"
                decoded_text = result.get("text")
                new_prefix = ""
                for segment in result["segments"]:
                    if segment["temperature"] < 0.5 and segment["no_speech_prob"] < 0.6:
                        new_prefix += segment["text"]
                    else:
                        # Clear history if the translation is unreliable, otherwise prompting on this leads to
                        # repetition and getting stuck.
                        clear_buffers = True

            previous_text.append(new_prefix)

            if clear_buffers or previous_text.has_repetition():
                audio_buffer.clear()
                previous_text.clear()

            translated_text = ''
            if translate_to_Chinese:
                # need_translate_text = decoded_text
                need_translate_text = tokenizer(decoded_text, return_tensors="pt")
                translation = translate_model.generate(**need_translate_text)
                translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
                # need_translate_text = need_translate_text.split('.')
                # translated_text = translator.translate(decoded_text, dest='zh-cn', src='en')

            # print(f'{datetime.now().strftime("%H:%M:%S")} {decoded_language} {decoded_text}')
            print(f'{datetime.now().strftime("%H:%M:%S")} {decoded_language} {decoded_text}\n\t   {translated_text}\n')

    finally:
        mic_stream.stop_stream()
        mic_stream.close()


def cli():
    parser = argparse.ArgumentParser(description="Parameters for translator.py")
    parser.add_argument('--model', type=str,
                        choices=['tiny', 'tiny.en','base' ,'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large'],
                        default='base',
                        help='Model to be used for generating audio transcription. Smaller models are faster and use '
                             'less VRAM, but are also less accurate. .en models are more accurate but only work on '
                             'English audio.')
    parser.add_argument('--language', type=str, default='auto',
                        help='Language spoken in the stream. Default option is to auto detect the spoken language. '
                             'See https://github.com/openai/whisper for available languages.')
    parser.add_argument('--interval', type=int, default=5,
                        help='Interval between calls to the language model in seconds.')
    parser.add_argument('--history_buffer_size', type=int, default=0,
                        help='Seconds of previous audio/text to use for conditioning the model. Set to 0 to just use '
                             'audio from the last interval. Note that this can easily lead to repetition/loops if the'
                             'chosen language/model settings do not produce good results to begin with.')
    parser.add_argument('--preferred_quality', type=str, default='audio_only',
                        help='Preferred stream quality option. "best" and "worst" should always be available. Type '
                             '"streamlink URL" in the console to see quality options for your URL.')
    parser.add_argument('--disable_vad', action='store_true',
                        help='Set this flag to disable additional voice activity detection by Silero VAD.')
    parser.add_argument('--use_faster_whisper', action='store_true',
                        help='Set this flag to use faster-whisper implementation instead of the original OpenAI '
                             'implementation.')
    parser.add_argument('--faster_whisper_model_path', type=str, default='whisper-large-v2-ct2/',
                        help='Path to a directory containing a Whisper model in the CTranslate2 format.')
    parser.add_argument('--faster_whisper_device', type=str, choices=['cuda', 'cpu', 'auto'], default='cuda',
                        help='Set the device to run faster-whisper on.')
    parser.add_argument('--faster_whisper_compute_type', type=str, choices=['int8', 'int8_float16', 'int16', 'float16'],
                        default='float16',
                        help='Set the quantization type for faster-whisper. See '
                             'https://opennmt.net/CTranslate2/quantization.html for more info.')
    parser.add_argument("--translate_to_Chinese", action='store_true',
                        help="Translate the recognized text into Chinese.")

    args = parser.parse_args().__dict__
    args["use_vad"] = not args.pop("disable_vad")
    use_faster_whisper = args.pop("use_faster_whisper")
    faster_whisper_args = dict()
    faster_whisper_args["model_path"] = args.pop("faster_whisper_model_path")
    faster_whisper_args["device"] = args.pop("faster_whisper_device")
    faster_whisper_args["compute_type"] = args.pop("faster_whisper_compute_type")

    if args['model'].endswith('.en'):
        if args['model'] == 'large.en':
            print("English model does not have large model, please choose from {tiny.en, small.en, medium.en}")
            sys.exit(0)
        if args['language'] != 'English' and args['language'] != 'en':
            if args['language'] == 'auto':
                print("Using .en model, setting language from auto to English")
                args['language'] = 'en'
            else:
                print("English model cannot be used to detect non-english language, please choose a non .en model")
                sys.exit(0)

    if args['language'] == 'auto':
        args['language'] = None

    # print(args['translate_to_Chinese'])
    # print(type(args['translate_to_Chinese']))
    # translate_to_Chinese_flag = args['translate_to_Chinese']
    # if args['beam_size'] == 0:
    #     args['beam_size'] = None

    main(device_index=0, faster_whisper_args=faster_whisper_args if use_faster_whisper else None, **args)
    # main(device_index=0,  faster_whisper_args=faster_whisper_args if use_faster_whisper else None, **args)


if __name__ == '__main__':
    cli()
