
from tempfile import NamedTemporaryFile
import torch
import gradio as gr
from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen


MODEL = None


def load_model():
    print("Loading model")
    return MusicGen.get_pretrained("melody")


def predict(texts, melodies):
    global MODEL
    if MODEL is None:
        MODEL = load_model()

    duration = 12
    MODEL.set_generation_params(duration=duration)

    print(texts, melodies)
    processed_melodies = []

    target_sr = 32000
    target_ac = 1
    for melody in melodies:
        if melody is None:
            processed_melodies.append(None)
        else:
            sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t()
            if melody.dim() == 1:
                melody = melody[None]
            melody = melody[..., :int(sr * duration)]
            melody = convert_audio(melody, sr, target_sr, target_ac)
            processed_melodies.append(melody)

    outputs = MODEL.generate_with_chroma(
        descriptions=texts,
        melody_wavs=processed_melodies,
        melody_sample_rate=target_sr,
        progress=False
    )

    outputs = outputs.detach().cpu().float()
    out_files = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(file.name, output, MODEL.sample_rate, strategy="loudness", add_suffix=False)
            waveform_video = gr.make_waveform(file.name)
            out_files.append(waveform_video)
    return [out_files]


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # CRESCENDO

        Welcome to CRESCENDO Audio Generation :)
        """
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                text = gr.Text(label="Describe your music", lines=2, interactive=True)
                melody = gr.Audio(source="upload", type="numpy", label="Condition on a melody (optional)", interactive=True)
            with gr.Row():
                submit = gr.Button("Generate")
        with gr.Column():
            output = gr.Video(label="Generated Music")
    submit.click(predict, inputs=[text, melody], outputs=[output], batch=True, max_batch_size=12)

    gr.Markdown("""
        ### More details 

        For More details Read CRESCENDO Docs ;)
        """)

demo.queue(max_size=15).launch()
