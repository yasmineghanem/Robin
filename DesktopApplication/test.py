import speech_recognition as sr

# continuous listen to speech and print it 
r = sr.Recognizer()
r.energy_threshold = 4000

m = sr.Microphone()

with m as source:
    # r.adjust_for_ambient_noise(source, duration=1)
    print("Say something!")
    audio = r.listen(source)
    
try:
    print("You said: " + r.recognize_google(audio))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
    
# continuous listen to speech and print it 
