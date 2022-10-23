import torch

from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
try:
  from TTS.utils.audio import AudioProcessor
except:
  from TTS.utils.audio import AudioProcessor


from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.models.vits import *

from config import *

MODEL_PATH = 'best_model.pth.tar'
CONFIG_PATH = 'config.json'
TTS_LANGUAGES = "language_ids.json"
TTS_SPEAKERS = "speakers.json"

class TTSInterface():

    def __init__(self,directory):
        self._directory = directory
        self._prepare_model()

    def _prepare_model(self):

        # load the config
        self.C = load_config(self._directory+"/config.json")

        # load the audio processor
        self.ap = AudioProcessor(**C.audio)

        self.speaker_embedding = None

        self.C.model_args['d_vector_file'] = self._directory+"/speakers.json"
        self.C.model_args['use_speaker_encoder_as_loss'] = False

        model = setup_model(C)
        model.language_manager.set_language_ids_from_file(self._directory+"/language_ids.json")
        # print(model.language_manager.num_languages, model.embedded_language_dim)
        # print(model.emb_l)
        cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        # remove speaker encoder
        model_weights = cp['model'].copy()
        for key in list(model_weights.keys()):
            if "speaker_encoder" in key:
                del model_weights[key]

        model.load_state_dict(model_weights)

        model.eval()

        if USE_CUDA:
            model = model.cuda()

        # synthesize voice
        self.use_griffin_lim = False
        self.speaker_manager= SpeakerManager(encoder_model_path=self._directory+"/SE_checkpoint.pth", encoder_config_path=self._directory+"/config_se.json",use_cuda=USE_CUDA)

    def compute_spec(self,y,samp=16000):
        spec=self.ap.spectogram(y,samp)
        spec = torch.FloatTensor(spec).unsqueeze(0)
        return spec
