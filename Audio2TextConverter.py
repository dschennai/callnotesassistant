import datetime
import json
import math
import os
import uuid
from pathlib import Path
import openai
import pandas as pd
import pydub
import requests
import streamlit as st
from nltk.tokenize import word_tokenize
from pydub import AudioSegment
from pydub.utils import make_chunks
import subprocess
# install ffmpeg and ffprobe using apt-get
subprocess.call(['apt-get', 'update'])
subprocess.call(['apt-get', 'install', '-y', 'ffmpeg'])

# Add your key and endpoint
openai.api_key=st.secrets["ds_org_key"]
discord_webhook = st.secrets["discord_webhook"]
azure_translator_key = st.secrets["azure_translator_key"]
endpoint = "https://api.cognitive.microsofttranslator.com/translate"


st.set_page_config(
    page_title="Audio Transcripts",
    page_icon="Logo_with_trademark.png",
    layout="wide"
)

def getuniquefilename():
    import datetime

    # Generate a unique number using the current date and time
    now = datetime.datetime.now()
    unique_number = now.strftime("%Y%m%d%H%M%S")
    return unique_number


def split_audio_file(audio_filename, segment_duration):
    input_file_path = 'audio/' + audio_filename
    output_directory_path = 'audio/'

    # Load the audio file
    audio = AudioSegment.from_wav(input_file_path)

    # Calculate the number of segments
    num_segments = len(audio) // segment_duration + 1
    splitaudionames = []
    # Split the audio file into segments
    for i in range(num_segments):
        start = i * segment_duration
        end = (i + 1) * segment_duration
        segment = audio[start:end]
        output_file_path = os.path.join(output_directory_path, f'{audio_filename}_segment_{i + 1}.wav')
        segment.export(output_file_path, format='wav')
        splitaudionames.append(output_file_path)
    print(f'{num_segments} segments created.')
    return splitaudionames


def translate_eng_tamil(englishtext):
    params = {
        'api-version': '3.0',
        'from': 'en',
        'to': ['ta']
    }
    headers = {
        'Ocp-Apim-Subscription-Key': azure_translator_key,
        # location required if you're using a multi-service or regional (not global) resource.
        'Ocp-Apim-Subscription-Region': "centralus",
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # You can pass more than one object in body.
    body = [{
        'text': englishtext
    }]

    request = requests.post(endpoint, params=params, headers=headers, json=body)
    response = request.json()
    # extract the translated text
    translated_text = response[0]['translations'][0]['text']

    # print the translated text
    return translated_text


def send_discord_webhook_message(webhook_url, message):
    payload = {'content': message}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
    if response.status_code == 204:
        print('Message sent successfully')
    else:
        print('Error sending Discord message: {}'.format(response.text))


def save_audio_files_to_folder(audiofile):
    filename_returned = "None"
    if audiofile is not None:
        if audiofile.name.endswith('wav'):
            audio = pydub.AudioSegment.from_wav(audiofile)
            file_type = 'wav'
            save_path = Path("audio") / audiofile.name
            audio.export(save_path, format=file_type)
            filename_returned = audiofile.name
        elif audiofile.name.endswith('mp3') or audiofile.name.endswith('mpga')or audiofile.name.endswith('m4a'):
            file_name_without_extension = os.path.splitext(audiofile.name)[0]
            file_name_without_extension = file_name_without_extension + ".wav"
            audio = pydub.AudioSegment.from_mp3(audiofile)
            file_type = 'mp3'
            audio.export("audio/" + file_name_without_extension, format="wav")
            filename_returned = file_name_without_extension
        elif audiofile.name.endswith("mp4"):
            file_name_without_extension = os.path.splitext(audiofile.name)[0]
            file_name_without_extension = file_name_without_extension + ".wav"
            audio = pydub.AudioSegment.from_file(audiofile, format="mp4")
            # file_type = 'mp3'
            # save_path = Path("audio") / audiofile.name
            audio.export("audio/" + file_name_without_extension, format="wav")

            filename_returned = file_name_without_extension
        elif audiofile.name.endswith('aac'):
            file_name_without_extension = os.path.splitext(audiofile.name)[0]
            file_name_without_extension = file_name_without_extension + ".wav"
            # load AAC file using pydub
            audio = AudioSegment.from_file(audiofile, format="aac")

            # export audio to WAV format using pydub
            audiofile = audio.export(file_name_without_extension, format="wav")
            filename_returned = file_name_without_extension
        elif audiofile.name.endswith('ogg'):
            file_name_without_extension = os.path.splitext(audiofile.name)[0]
            file_name_without_extension = file_name_without_extension + ".wav"

            audio = AudioSegment.from_file(audiofile, format="ogg")
            audiofile = audio.export("audio/" + file_name_without_extension, format="wav")
            filename_returned = file_name_without_extension
    print(filename_returned)

    return filename_returned, audio


def upload_audio_to_folder(audiofile):
    if audiofile is not None:
        # get the file extension
        extension = Path(audiofile.name).suffix.lower()

        # determine the audio format
        if extension == ".wav":
            audio = AudioSegment.from_file(audiofile, format=extension)
        elif extension == ".mp3":
            audio = AudioSegment.from_file(audiofile, format=extension)
        elif extension == ".aac":
            audio = AudioSegment.from_file(audiofile, format=extension)
        elif extension == ".ogg":
            audio = AudioSegment.from_file(audiofile, format=extension)
        else:
            raise ValueError("Unsupported file format")

        # set the output file path and format
        output_path = Path("audio") / Path(audiofile.name).stem
        output_format = "wav"

        # export the audio file
        audio.export(output_path.with_suffix(f".{output_format}"), format=output_format)

        # return the output file name and audio segment
        return output_path.name + "." + output_format, audio
    else:
        return "None", None


def transcribe_audio(file_name_alone):
    # append the folder name "audio"
    model_id = 'whisper-1'
    # media_file = open("audio/" + file_name_alone, 'rb')
    media_file = open(file_name_alone, 'rb')
    # start_time = time.time()  # remove in case do not want to count time
    response = openai.Audio.transcribe(
        model=model_id,
        file=media_file)
    return response["text"]

def usewhisperapi(filename):
    import whisper

    model = whisper.load_model("base")
    result = model.transcribe(filename)
    print(result["text"])


def openai_translate_output_text_english(file_name_alone):
    # append the folder name "audio"
    model_id = 'whisper-1'

    # media_file = open("audio/" + file_name_alone, 'rb')
    media_file = open(file_name_alone, 'rb')

    response = openai.Audio.translate(
        model=model_id,
        file=media_file
    )
    return response["text"]


def break_up_file(tokens, chunk_size, overlap_size):
    if len(tokens) <= chunk_size:
        yield tokens
    else:
        chunk = tokens[:chunk_size]
        yield chunk
        yield from break_up_file(tokens[chunk_size - overlap_size:], chunk_size, overlap_size)


def break_up_file_to_chunks(filename, chunk_size=2000, overlap_size=100):
    with open(filename, 'r') as f:
        text = f.read()
    tokens = word_tokenize(text)
    return list(break_up_file(tokens, chunk_size, overlap_size))


def convert_to_prompt_text(tokenized_text):
    prompt_text = " ".join(tokenized_text)
    prompt_text = prompt_text.replace(" 's", "'s")
    return prompt_text


def get_meeting_mintues_da_vinci(calldetail):
    filename = "callnotes.txt"
    prompt_response = []

    chunks = break_up_file_to_chunks(filename)
    for i, chunk in enumerate(chunks):
        prompt_request = call_detail + "Summarize this meeting transcript: " + convert_to_prompt_text(chunks[i])
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt_request,
            temperature=.5,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        prompt_response.append(response["choices"][0]["text"].strip())

    prompt_request = "Consolidate these meeting summaries: " + prompt_response

    summaryresponse = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_request,
        temperature=.5,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    meeting_summary = summaryresponse["choices"][0]["text"].strip()
    return str(prompt_response), meeting_summary


def get_action_items_gpt35turbo():
    action_response = []
    filename = "callnotes.txt"

    chunks = break_up_file_to_chunks(filename)
    for i, chunk in enumerate(chunks):
        prompt_request = "Provide a list of action items with a due date from the provided meeting transcript text: " + convert_to_prompt_text(
            chunks[i])
        messages = [{"role": "system", "content": "This is text summarization."}]
        messages.append({"role": "user", "content": prompt_request})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=.5,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        action_response.append(response["choices"][0]["message"]['content'].strip())

    prompt_request = "Consoloidate these meeting action items: " + str(action_response)
    messages = [{"role": "system", "content": "This is text summarization."}]
    messages.append({"role": "user", "content": prompt_request})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=.5,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    meeting_action_items = response["choices"][0]["message"]['content'].strip()
    return str(action_response), meeting_action_items


def get_action_items_davinci():
    action_response = []
    filename = "callnotes.txt"
    chunks = break_up_file_to_chunks(filename)
    for i, chunk in enumerate(chunks):
        prompt_request = "Provide a list of action items with a due date from the provided meeting transcript text: " + convert_to_prompt_text(
            chunks[i])

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt_request,
            temperature=.5,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        action_response.append(response["choices"][0]["text"].strip())
        action_response.append("\n")
    prompt_request = "Consoloidate these meeting action items: " + str(action_response)

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_request,
        temperature=.5,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    meeting_action_items = response["choices"][0]["text"].strip()

    return str(action_response), meeting_action_items

def get_notes_cleanedup(call_detail):
    filename = "callnotes.txt"
    prompt_response = []
    chunks = break_up_file_to_chunks(filename)
    main_command= """ Please follow the instructions below while working on the transcript:
Edit repetitions: If a speaker repeats a word or phrase multiple times, please edit it down to a single occurrence. 
Maintain meaning and context: While editing, ensure that the meaning and context of the conversation remain intact. Do not remove any content that is relevant to the discussion or alters the intended message.
Improve sentence structure and grammar: In addition to removing filler words, please proofread the transcript for grammatical errors and awkward sentence structures.
Preserve natural speech: Although the goal is to create a polished transcript, it's essential to maintain a conversational tone"""

    #messages.append({"role": "user", "content": main_command})

    for i, chunk in enumerate(chunks):
        prompt_request = convert_to_prompt_text(chunks[i])
        messages = [{"role": "system","content": "i will present a meeting transcript. Format the transcript in proper sentences"}]
        messages.append({"role": "user", "content": "Hello, welcome to my lecture."})
        messages.append({"role": "user", "content": main_command})
        messages.append({"role": "user", "content": prompt_request})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=.5,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    return response["choices"][0]["message"]['content'].strip()

def get_meeting_mintues_gpt35turbo(call_detail):
    print("starting..")
    filename = "callnotes.txt"
    prompt_response = []

    chunks = break_up_file_to_chunks(filename)
    for i, chunk in enumerate(chunks):
        prompt_request = call_detail + "Summarize this meeting transcript : " + convert_to_prompt_text(chunks[i])

        messages = [{"role": "system", "content": "This is text summarization."}]
        messages.append({"role": "user", "content": prompt_request})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=.5,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        prompt_response.append(response["choices"][0]["message"]['content'].strip())

    prompt_request = "Consoloidate these meeting summaries: " + str(prompt_response)
    messages = [{"role": "system", "content": "This is text summarization."}]
    messages.append({"role": "user", "content": prompt_request})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=.5,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    meeting_summary = response["choices"][0]["message"]['content'].strip()
    print(meeting_summary)
    return str(prompt_response), meeting_summary

def split_audio_files_by_size(audio_data, audiof_filename):
    filename_wo_extn = os.path.splitext(audiof_filename)[0]
    channel_count = audio_data.channels  # Get channels
    sample_width = audio_data.sample_width  # Get sample width
    duration_in_sec = len(audio_data) / 1000  # Length of audio in sec
    sample_rate = audio_data.frame_rate
    bit_rate = 16  # assumption , you can extract from mediainfo("test.wav") dynamically
    wav_file_size = (sample_rate * bit_rate * channel_count * duration_in_sec) / 8
    print(wav_file_size)
    file_split_size = 19000000  # 10Mb OR 10, 000, 000 bytes
    total_chunks = wav_file_size // file_split_size
    print(total_chunks)
    # Get chunk size by following method #There are more than one ofcourse
    # for  duration_in_sec (X) -->  wav_file_size (Y)
    # So   whats duration in sec  (K) --> for file size of 10Mb
    #  K = X * 10Mb / Y

    chunk_length_in_sec = math.ceil((duration_in_sec * file_split_size) / wav_file_size)  # in sec
    chunk_length_ms = chunk_length_in_sec * 1000
    chunks = make_chunks(audio_data, chunk_length_ms)

    # Export all of the individual chunks as wav files
    audio_segments_list = []
    for i, chunk in enumerate(chunks):
        chunk_name = "audio/" + filename_wo_extn + "chunk{0}.wav".format(i)
        chunk.export(chunk_name, format="wav")
        audio_segments_list.append(chunk_name)
    print("File Split Completed")
    return audio_segments_list

def millisec(timeStr):
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s

def startprocessing():
    with st.spinner('Wait for it...'):
        outputtext =""
        outputtextinenglish = ""

        # split large files to smaller chunks
        audiosegmentsforopenai = split_audio_files_by_size(audio_data, save_audio_file_name)
        for audiosegs in audiosegmentsforopenai:        #
            #outputtext = outputtext + transcribe_audio(audiosegs)+ "\n\n"
            outputtextinenglish = outputtextinenglish + openai_translate_output_text_english(audiosegs)+"\n\n"

        if sel_lang == "Tamil":
            azuretranslatedtext_mixtamil = translate_eng_tamil(outputtextinenglish)
        else:
            azuretranslatedtext_mixtamil = "Not Translated : As per User Input"

        translatetab, azuretamil = st.tabs(["English Version", "Tamil Version"])
        #with transcribetab:
        #    st.text_area("ssa", outputtext, label_visibility="collapsed")
        #    st.download_button(label="Download Text", data=outputtext,
        #                       file_name=save_audio_file_name + ".txt", mime="text/plain")
        with translatetab:
            st.download_button(label="Download", data=outputtextinenglish,
                               file_name=save_audio_file_name + ".txt", mime="text/plain")
            st.text_area("sa", outputtextinenglish ,label_visibility="collapsed", height=250)

        with azuretamil:
            if azuretranslatedtext_mixtamil == 'Not Translated : As per User Input':
                st.text_area("sad", azuretranslatedtext_mixtamil, label_visibility="collapsed", height=250)
            else:
                st.download_button(label="Download", data=azuretranslatedtext_tamil,
                               file_name=save_audio_file_name + ".txt", mime="text/plain")
                st.text_area("sad", azuretranslatedtext_mixtamil, label_visibility="collapsed", height=250)

        # Following stores output in txt file
        with open(getuniquefilename() + "_transcribedoutput.txt", "w") as f:
            f.write(outputtextinenglish)


        if discord_message == 'Yes' and sel_lang == "Tamil":
            if len(azuretranslatedtext_mixtamil) < 2000:
                send_discord_webhook_message(discord_webhook, azuretranslatedtext_mixtamil)
        else:
            if len(outputtextinenglish) < 2000:
                send_discord_webhook_message(discord_webhook, outputtextinenglish)

st.subheader("Audio2Text Assistant")
font_css = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 18px;font-style: bold;
}
</style>
"""
st.markdown('<style>body{background-color: Blue;}</style>', unsafe_allow_html=True)
st.write(font_css, unsafe_allow_html=True)

st.sidebar.write("About Audio2Text Tool")
st.sidebar.write("Supports mp3, ogg, acc file formats")
st.sidebar.write("Use the options provided to choose the Desired Output Language")

tab1, tab2, tab3 = st.tabs(["Convert Audio to Text", "Optimize The Summary", "Get Minutes of Meeting"])
with tab1:
    form = st.form(key="annotation", clear_on_submit=False)
    col1, col2 = st.columns([3, 1])
    with col1:
        with form:
            cols = st.columns((2.5, 2, 2))
            cols[0].write("Audio Description/Title")
            audio_description = cols[0].text_input("Audio Title",
                                                   help="Add Client Name, if applicable. Eg - Weekly Call with JOGO, REAN Meeting Minutes, Web Design Team call",
                                                   placeholder="Weekly Call with JOGO, Team Leads Call",
                                                   label_visibility="collapsed")

            cols[0].write("Upload your Audio File ( Files larger than 10MB take more processing time)")
            uploaded_file = cols[0].file_uploader("Upload Recording",
                                                  type=['wav', 'mp3', 'aac', 'ogg', 'mpga', 'mp4','m4a'],
                                                  accept_multiple_files=False, key="u1", label_visibility="collapsed")
            if uploaded_file is not None:
                with st.spinner('Wait for it...'):
                    save_audio_file_name, audio_data = save_audio_files_to_folder(uploaded_file)
                    # save_audio_file_name, audio_data = upload_audio_to_folder(uploaded_file)

                    # get the duration of the audio file in seconds
                    duration_in_seconds = len(audio_data) / 1000

                    # format the duration as minutes and seconds
                    duration_in_minutes = int(duration_in_seconds // 60)
                    duration_in_seconds = int(duration_in_seconds % 60)
                    # format the file size as MB with one decimal place
                    file_size_in_mb = uploaded_file.size / 1024 / 1024
                    file_size_formatted = f"{file_size_in_mb:.1f} MB"

                    # display the file name, file size, and duration to the user
                    # new_title = f'<p style="font-family:sans-serif; color:Blue; font-size: 14px;">Audio Size : {file_size_formatted}  / Audio Duration :{duration_in_minutes} minutes, {duration_in_seconds} seconds</p>'
                    # st.markdown(new_title, unsafe_allow_html=True)

            # cols[0].info("""It takes approximately 40 seconds to process 15 minutes of audio.""")

            cols[1].write("Desired Output Language")
            sel_lang = cols[1].radio(
                "Language", ["Same as Input", "English", "Tamil"], label_visibility="collapsed", index=0,
                horizontal=True)

            cols[1].write("Select Meeting Type")
            meetingmode = cols[1].selectbox(
                "Meeting", ["Recording", "Zoom", "Phone", "In Person"], label_visibility="collapsed",
                key="meetmode")

            #cols[1].write("Multiple Speaker Identification Required")
            #speaker_identification = cols[1].radio("Multiple Speaker Identification Required", ["On", "Off"], index=1,
            #                                       horizontal=True, label_visibility="collapsed")

            cols[2].write("Date and Time of Meeting")
            s1, s2, s3 = cols[2].columns([0.45, 0.45, 0.5])
            audiodate = s1.date_input("Date and Time of Meeting", label_visibility="collapsed")
            audiotime: object = s2.time_input("Time", datetime.time(12, 0), label_visibility="collapsed")

            cols[2].write("Send Transcript to Discord Channel")
            discord_message = cols[2].radio("Send Transcript to Discord Channel:", ["Yes", "No"], index=1,
                                            horizontal=True,
                                            label_visibility="collapsed")

            #new_title = f'<p style="font-family:sans-serif; color:Blue; font-size: 14px;">Does this audio contain languages other than English (ignore if its just a couple of words in other languages)</p>'
            #cols[2].markdown(new_title, unsafe_allow_html=True)
            ## cols[2].write("Does this audio contain languages other than English (ignore if its just a couple of words in other languages)")
            #single_lang = cols[2].radio("single_lang", ["Yes", "No"], horizontal=True, index=1,
            #                            label_visibility="collapsed")

            submit_button = st.form_submit_button(label="  Submit ")

    if submit_button:
        status_flag = False
        # Check the values entered
        if not uploaded_file:
            st.warning("No Audio Files Uploaded Yet.")
            status_flag = False
        elif audio_description == '':
            st.warning("Please enter a valid Audio Description.")
            status_flag = False

        if uploaded_file and len(audio_description) > 0:
            startprocessing()

with tab3:
    st.write("Context/ Title for the Notes (eg) REAN Weekly Call on AHA Video or Call with Lisa and SK")
    call_detail = st.text_input("summary", "", label_visibility="collapsed")

    #cols[1].write("Fine tune using Da Vinci(basic) or GPT 3.5")
    #finetune = cols[1].radio("tune", ["Da Vinci 003", "GPT 3.5"], index=1, horizontal=True,
    #                         label_visibility="collapsed")
    st.write("Add your Call Notes")
    callnotes = st.text_area("tab2_call", "", label_visibility="collapsed", height=100)

    if st.button("Get Meeting Minutes with Action Items") and len(callnotes) > 0:
        file = open("callnotes.txt", "w")
        file.write(callnotes)
        file.close()

        #if finetune == "GPT 3.5":
        st.write("")
        plain_response, meeting_summary = get_meeting_mintues_gpt35turbo(callnotes)
        actionitem_response, consolidated_actionitem = get_action_items_gpt35turbo()
        t2, t4 = st.tabs(["Meeting summary", "Consolidated Action Items"])
        #with t1:
        #    st.write(plain_response)
        with t2:
            st.text_area("meeting_summary",meeting_summary, label_visibility="collapsed", height=250)
        #with t3:
        #    st.write(actionitem_response)
        with t4:
            st.text_area("consolidated_actionitem", consolidated_actionitem, label_visibility="collapsed", height=250)

with tab2:
    st.write("Add your Call Notes")
    callnotes = st.text_area("input", "", label_visibility="collapsed", height=100)

    if st.button("Refine the Meeting Notes") and len(callnotes) > 0:
        with st.spinner('Wait for it...'):
            file = open("callnotes.txt", "w")
            file.write(callnotes)
            file.close()

            #if finetune == "GPT 3.5":
            st.subheader("Meeting Summary Refined using GPT3.5")
            plain_response = get_notes_cleanedup(callnotes)
            st.text_area("output",plain_response, label_visibility="collapsed", height=300)
            st.markdown("---")
