import keyboard
'''k = keyboard.read_key()  # in my python interpreter, this captures "enter up"
k = keyboard.read_key()'''
recorded = keyboard.record(until='esc')
# Then replay back at three times the speed.
print(recorded)