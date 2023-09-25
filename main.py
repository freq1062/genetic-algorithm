from music21 import converter, midi, note, chord, pitch
import fractions
from itertools import permutations
import numpy as np
from random import choice, randint, randrange, random, choices
from tqdm.auto import tqdm
import collections
import pandas as pd
from copy import copy
import matplotlib.pyplot as plt
import pretty_midi

def constSetup():
    pitches, durations, beats = collections.defaultdict(list),collections.defaultdict(list),collections.defaultdict(list)
    permut = ['10000','11000', '11100', '11110','11111'] #Encode notes in permutations
    for c in permut:
        for p in permutations(c):
            if ''.join(p) not in pitches["Key"]:
                pitches['Key'].append(str(''.join(p)))
    #32 combinations, #00000 is rest
    scale = ["C", "D", "E", "F", "G", "A", "B"] #All possible notes are in this scale
    for i in range(0, len(pitches['Key'])):
        pitches['Pitches'].append(pitch.Pitch(scale[i%len(scale)]+str(int(np.ceil(i/len(scale)) +2 ))).midi) #Convert note to midi number
    durations["QuarterValue"].extend([0.5, 1, 2, 4]) #eigth, quarter, half, whole notes
    durations["Key"].extend(["00", "01", "10", "11"])
    beats["Value"].extend(list(range(0,16))) #8 beats in a bar, each beat is 1/2 a quarter note long
    permut = ['0000', '1000', '1100', '1110', '1111']
    for c in permut:
        for p in permutations(c):
            if ''.join(p) not in beats["Key"]:
                beats['Key'].append(str(''.join(p)))
    return pitches,durations,beats

def open_midi(filename=str): #Gets the midi data from a filename in the directory
    # There is an one-line method to read MIDIs but to remove the drums we need to manipulate some low level MIDI events.
    mf = midi.MidiFile()
    mf.open(filename)
    mf.read()
    mf.close()
    return midi.translate.midiFileToStream(mf)

def notes_to_midi( #Convert note data to midi file
  notes: pd.DataFrame,
  out_file: str, 
  instrument_name: str,
  velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:
  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))
  for i, note in notes.iterrows():
    start = float(note['beat'])#float(prev_start + note['step'])
    end = float(start + float(note['duration']))
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm

def closest(lst, K):return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def extract_notes(midi_part, key): #Gets an array of all notes from a midi file: [["pitch": [float midi number] ["beat": float 1-4] ["duration": float length in quarter notes] ["bar": int bar of note of same index]]]
    #Assuming 4/4 time signature
    notes = collections.defaultdict(list)
    prevBeat = 0
    barCounter = 0
    barNotes = []
    if "major" in key: key = key[0] 
    else: key = key[0].lower()
    print("Extracting notes from file")
    for nt in tqdm(midi_part.flatten().notes[:]):
        if isinstance(nt, note.Note):
            notes['pitch'].append(closest(PITCHES['Pitches'],nt.pitch.midi))
            if isinstance(nt.beat, fractions.Fraction):
                notes['beat'].append(closest(BEATS["Value"], (nt.beat.numerator/nt.beat.denominator-1)*4 ))
            else:
                notes['beat'].append(closest(BEATS["Value"], (nt.beat-1)*4))
            if isinstance(nt.duration.quarterLength, fractions.Fraction):
                notes['duration'].append(closest(DURATIONS["QuarterValue"],round(((nt.duration.quarterLength.numerator/nt.duration.quarterLength.denominator *2)) /2)))
            else:
                notes['duration'].append(closest(DURATIONS["QuarterValue"], nt.duration.quarterLength))
            if prevBeat > nt.beat:
                barNotes = []
                barNotes.append(str(nt.pitch))
                barCounter+=1
            else: barNotes.append(str(nt.pitch))
            notes["bar"].append(barCounter)
            prevBeat = nt.beat
        elif isinstance(nt, chord.Chord):
            for pitch in nt.pitches:
                notes['pitch'].append(closest(PITCHES['Pitches'],pitch.midi))
                if isinstance(nt.beat, fractions.Fraction):
                    notes['beat'].append(closest(BEATS["Value"], (nt.beat.numerator/nt.beat.denominator-1)*4 ))
                else:
                    notes['beat'].append(closest(BEATS["Value"], nt.beat-1)*4)
                if isinstance(nt.duration.quarterLength, fractions.Fraction):
                    notes['duration'].append(closest(DURATIONS["QuarterValue"],round(((nt.duration.quarterLength.numerator/nt.duration.quarterLength.denominator *2)) /2)))
                else:
                    notes['duration'].append(closest(DURATIONS["QuarterValue"], nt.duration.quarterLength))
                if prevBeat > nt.beat:
                    barNotes = []
                    barNotes.append(str(pitch))
                    barCounter+=1
                else: barNotes.append(str(pitch))
                notes["bar"].append(barCounter)
                prevBeat = nt.beat
    return notes

def MelodyToGenome(notes): #Converts note data into a genome
    genome = ""
    noteData = copy(notes)
    for i in range(len(BEATS["Value"])): #number of beats in a bar
        for j in range(BEAT_NOTES): #number of notes per beat allowed at the same time
            if float(i) in noteData['beat']:
                currentIndex = noteData['beat'].index(float(i))
                genome=genome+PITCHES["Key"][PITCHES['Pitches'].index(noteData['pitch'][currentIndex])]
                genome=genome+DURATIONS['Key'][DURATIONS['QuarterValue'].index(min(DURATIONS['QuarterValue'], key=lambda x:abs(float(x)-float(noteData['duration'][currentIndex]))))]
                genome=genome+BEATS['Key'][BEATS['Value'].index(int(i))]
                noteData["beat"][currentIndex] = -1
            else:
                genome=genome+"00000"
                genome=genome+DURATIONS['Key'][DURATIONS['QuarterValue'].index(0.5)]
                genome=genome+BEATS['Key'][BEATS['Value'].index(int(i))]
    return genome

def GenomeToMelody(genome=str): #Converts a genome
    notes = [genome[i:i+ (len(BEATS['Key'][0])+len(DURATIONS['Key'][0])+len(PITCHES['Key'][0])) ] for i in range(0, len(genome), (len(BEATS['Key'][0])+len(DURATIONS['Key'][0])+len(PITCHES['Key'][0])))]
    melody = collections.defaultdict(list)
    for ele in notes:
        if ele[:5] != "00000": #Disregard rests
            melody["pitch"].append(PITCHES['Pitches'][PITCHES['Key'].index(ele[:5])])
            melody['duration'].append(DURATIONS["QuarterValue"][DURATIONS['Key'].index(ele[5:7])])
            melody['beat'].append(BEATS['Value'][BEATS['Key'].index(ele[7:])])
    return melody

def z_score(value, arr): #Zscore 0: Exactly the mean; greater the value the farther it is
    std = np.std(arr)
    return abs(value - sum(arr)/len(arr))/std

def scale(target, predicted, valueArr=[]):
    if target-predicted == 0 : return 0
    return abs(target-predicted)**2

def fitnessFunction(notesOriginal, notesGenerated, barO=int):
    score = 0
    notesO = collections.defaultdict(list) #Note data of original song
    notesG = copy(notesGenerated) #Note data of generated genome
    for i in range(0, len(notesOriginal['bar'])): #Shorten notesO to only the notes in barO
        if int(notesOriginal['bar'][i]) == barO:
            notesO["pitch"].append(notesOriginal["pitch"][i])
            notesO["duration"].append(notesOriginal["duration"][i])
            notesO["beat"].append(notesOriginal["beat"][i])
    for j in range(0, len(notesO['duration'])):#Make sure that all the notes in the non-genome file fit with the beat system
        notesO["duration"][j] = min(DURATIONS['QuarterValue'], key=lambda x:abs(float(x)-float(notesO["duration"][j])))
        notesO["beat"][j] = min(BEATS['Value'], key=lambda x:abs(float(x)-(float(notesO["beat"][j]))))
    maxS = (scale(max(PITCHES['Pitches']),min(PITCHES['Pitches'])) + scale(max(DURATIONS['QuarterValue']),min(DURATIONS['QuarterValue'])) + scale(max(BEATS['Value']),min(BEATS['Value'])))
    cIndexes = [] #Index of all the notes in notesG that have already been taken
    for k in range(len(notesO['pitch'])):
        cScores = []
        if len(notesG['pitch']) == len(cIndexes): 
            score += abs((len(notesO['pitch']) - len(cIndexes))) * maxS
            break
        for l in range(len(notesG['pitch'])):
            if l not in cIndexes:
                cScores.append(scale(notesO['pitch'][k],notesG['pitch'][l],PITCHES['Pitches']) * pitchFactor + scale(notesO['beat'][k], notesG['beat'][l],BEATS["Value"]))
            else:cScores.append(scale(max(PITCHES['Pitches']),min(PITCHES['Pitches'])) + scale(max(BEATS['Value']),min(BEATS['Value'])))#cScores.append(3+pitchFactor)
        ind = cScores.index(min(cScores))
        cIndexes.append(ind)
        score += min(cScores) + scale(notesO['duration'][k],notesG['duration'][ind],DURATIONS["QuarterValue"])
    if len(notesG['pitch'])>len(cIndexes): 
        score += abs((len(notesG['pitch']) - len(cIndexes))) * maxS
    return score

def generate_weighted_distribution(population,fitness_func, templateBar=int): #Based on how well each genome in a population scores it is more likely to be selected to the next generation this func generates the probabilities
    result = []
    for gene in population: result.append(fitness_func(extracted_template, GenomeToMelody(gene), templateBar+1))      
    return result

def selection_pair(population, fitness_func, templateBar):# Based on weighted distribution pick 2 from population to go to next generation
    distribution = generate_weighted_distribution(population, fitness_func, templateBar)
    return choices(population, weights=distribution, k=2)

def crossover(a:str, b:str):
    a = [a[i:i+(len(BEATS['Key'][0])+len(DURATIONS['Key'][0])+len(PITCHES['Key'][0]))] for i in range(0, len(a), (len(BEATS['Key'][0])+len(DURATIONS['Key'][0])+len(PITCHES['Key'][0])))]
    b = [b[i:i+(len(BEATS['Key'][0])+len(DURATIONS['Key'][0])+len(PITCHES['Key'][0]))] for i in range(0, len(b), (len(BEATS['Key'][0])+len(DURATIONS['Key'][0])+len(PITCHES['Key'][0])))]
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of same length")
    length = len(a)
    if length < 2:
        return a, b
    c1 = ""
    c2 = ""
    for _ in range(len(a)):
        p1 = randint(1, length - 1)
        p2 = randint(1, length - 1)
        c1+=choice([a,b])[p1]
        c2+=choice([a,b])[p2]
    return c1, c2

def mutation(genome: str, num: int = 50, probability: float = 0.5):# Probability is the chance that a note changes
    notes = [genome[i:i+(len(BEATS['Key'][0])+len(DURATIONS['Key'][0])+len(PITCHES['Key'][0]))] for i in range(0, len(genome), (len(BEATS['Key'][0])+len(DURATIONS['Key'][0])+len(PITCHES['Key'][0])))]
    for _ in range(num):
        index = randrange(len(notes))
        if random() > probability:
            if random() >= 0.5:
                notes[index].replace(notes[index][:5], choice(PITCHES["Key"]))
                notes[index].replace(notes[index][5:7], choice(DURATIONS["Key"]))
                notes[index].replace(notes[index][7:], choice(BEATS['Key']))
            else:
                notes[index].replace(notes[index][:5], "00000")
                notes[index].replace(notes[index][5:7], "01")
                notes[index].replace(notes[index][7:], choice(BEATS['Key']))
    notes = GenomeToMelody(''.join(notes))
    for beat in BEATS['Value']:
        if notes["beat"].count(beat) >3:
            for _ in range(notes["beat"].count(beat) - BEAT_NOTES):
                chosen = choice(list(np.where(np.array(notes['beat']) == beat)[0]))
                del notes["pitch"][chosen]
                del notes["beat"][chosen]
                del notes["duration"][chosen]
    return MelodyToGenome(notes)

def run_evolution(population=list,templateBar=int,generation_limit: int = 100,showMetrics=bool): #Run evolution for generation_limit number of generations
    #Metrics
    scores = []
    topScores = []
    avgScores = []
    botScores = []
    scorePlot = []
    aboveAscores = []
    avgPitches = []
    topPitches = []
    #Final output genome
    topGenome = choice(population)
    botGenome = choice(population)
    if showMetrics: print("Starting simulation on bar",templateBar+1)
    for _ in range(generation_limit): #Run the simulation
        scores = []
        avgPitch = 0
        #Sort each genome by score; best score first, worst last
        population = sorted(population, key=lambda genome: fitnessFunction(extracted_template, GenomeToMelody(genome), templateBar))#, reverse=True)
        #Collect score metrics
        for genome in population:
            scores.append(fitnessFunction(extracted_template, GenomeToMelody(genome), templateBar))
            avgPitch += len(GenomeToMelody(genome)['pitch'])
        avgPitches.append(avgPitch/len(population))
        #Select the genomes which are above average score for that generation to move on to next generation
        kNum = round( len([ a for a in scores if a >= (sum(scores)/len(scores)) ]) / 2 ) * 2
        next_generation = population[0:kNum]
        aboveAscores.append(list(map(lambda genome: fitnessFunction(extracted_template, GenomeToMelody(genome), templateBar) , next_generation)))
        #For remaining population conduct weighted selection, survivors crossover genes & mutate
        for j in range(int(len(population) / 2) - int(kNum/2) ):#int(len(population) / 3) - 6):
            parents = selection_pair(population, fitnessFunction, templateBar)
            offspring_a, offspring_b = crossover(parents[0], parents[1])
            offspring_a = mutation(offspring_a)
            offspring_b = mutation(offspring_b)
            next_generation += [offspring_a, offspring_b]
        #Collect score metrics
        botScores.append(max(scores))
        topScores.append(min(scores))
        avgScores.append(sum(scores)/len(scores))
        scorePlot.append(list(map(lambda genome: fitnessFunction(extracted_template, GenomeToMelody(genome), templateBar) , next_generation)))
        population = next_generation
        #If there is a new best genome keep track of it
        if min(scores) < (fitnessFunction(extracted_template, GenomeToMelody(topGenome), templateBar)): 
            topGenome = population[scores.index(min(scores))]
        topPitches.append(len(GenomeToMelody(topGenome)['pitch']))
        if max(scores) > fitnessFunction(extracted_template, GenomeToMelody(botGenome), templateBar): botGenome = population[scores.index(max(scores))]
    if showMetrics: 
        #print(botScores)
    #Plot collected data across generations
        plt.figure()
        #For small numbers of generations: Plot all scores in population & elitist accepted scores
        
        #for gen in range(0, len(scorePlot)):
        #    for score in scorePlot[gen]:
        #        plt.plot(gen, score, "r+")
        #for gen in range(0, len(aboveAscores)):
        #    for score in aboveAscores[gen]:
        #        plt.plot(gen, score, 'g+')
        plt.plot(list(range(0, generation_limit)), topScores, label="Top Score")
        plt.plot(list(range(0, generation_limit)), avgScores, label="Avg Score")
        plt.plot(list(range(0, generation_limit)), botScores, label="Worst Score")
        plt.legend(loc='upper right')
        plt.ylabel("Genome Score")
        plt.xlabel("Generation")
        plt.title("Evolution Progress")
        plt.figure()
        plt.plot(list(range(0, generation_limit)), avgPitches, label= "Avg # of Notes")
        plt.plot(list(range(0, generation_limit)), topPitches, label = "Best # of Notes")
        plt.legend(loc='upper right')
        plt.ylabel("Average Number of Notes per Genome")
        plt.xlabel("Generation")
        plt.title("Number of Notes Trend")
        plt.show()
    if showMetrics:
        print(pd.DataFrame(GenomeToMelody(topGenome)))
        print("Best genome score imitating bar",templateBar+1,":",str(fitnessFunction(extracted_template, GenomeToMelody(topGenome), templateBar)))
    return topGenome, fitnessFunction(extracted_template, GenomeToMelody(topGenome), templateBar)

def fillPopulation(numGenomes=int, barStart=int): #Create an array of randomly generated note data to be put into the simulation
    population = []
    numNotes = []
    for _ in range(0, numGenomes): 
        genomeNotes = collections.defaultdict(list)
        for _ in range(0, extracted_template['bar'].count(barStart) + round(10*random())): #Based on how many notes are in the requested bar to copy input this many random notes to speed up evolution
            genomeNotes['pitch'].append(choice(PITCHES["Pitches"]))
            genomeNotes['duration'].append(choice(DURATIONS["QuarterValue"]))
            genomeNotes['beat'].append(choice(BEATS["Value"]))
        population.append(MelodyToGenome(genomeNotes))
        numNotes.append(len(genomeNotes['pitch']))
    return population

def saveFile(name=str,notes=pd.DataFrame):
    out_pm = notes_to_midi(pd.DataFrame(notes), out_file=name, instrument_name="Acoustic Grand Piano")
    out_pm.write(name)
    print("Finished writing notes to file")

def generate(numBars=int, barStart=int, numGenerations=int,showMetrics=bool): #Oversee creation of population and evolution simulation to come up with the generated note data
    generatedNotes = []
    generatedNotes = pd.DataFrame(generatedNotes, columns=['pitch','duration','beat'])
    populationNotes = []
    bar = []
    for i in range(barStart, numBars+barStart): #On paper it's bar 1 but computer starts at bar 0
        bar = []
        populationNotes = fillPopulation(POPULATION_SIZE, barStart)
        populationNotes, score = run_evolution(populationNotes, i, numGenerations,showMetrics)
        print(populationNotes)
        print("Score:",score)
        bar = (GenomeToMelody(populationNotes))
        for j in range(0, len(bar['beat'])):
            bar['beat'][j] = (bar['beat'][j])/2
            bar['beat'][j] += ((i-barStart))
        bar = pd.DataFrame(bar)
        generatedNotes = pd.concat([generatedNotes, bar])
    return generatedNotes

def thresholdGenerate(numBars=int, barStart=int, numGenerations=int,showMetrics=bool,threshold=int,limit=int): #Oversee creation of population and evolution simulation to come up with the generated note data
    generatedNotes = []
    generatedNotes = pd.DataFrame(generatedNotes, columns=['pitch','duration','beat'])
    populationNotes = []
    bar = []
    statistics = []
    for i in tqdm(range(barStart, numBars+barStart)): #On paper it's bar 1 but computer starts at bar 0
        similarities = []
        for k in range(0, limit):
            populationNotes = fillPopulation(POPULATION_SIZE, barStart)
            populationNotes, score = run_evolution(populationNotes, i, numGenerations,showMetrics)
            similarities.append(score)
            if similarities[k] <= min(similarities):
                print(str(k)+"/"+str(limit)+" New Best: "+str(min(similarities)))
                bar = (GenomeToMelody(populationNotes))
            if min(similarities) <= threshold:
                bar = (GenomeToMelody(populationNotes))
                break
            if k == limit-1:
                statistics.append("\nBar: "+str(i)+" Best Score: "+str(min(similarities)))
        for j in range(0, len(bar['beat'])):
            bar['beat'][j] = (bar['beat'][j])/2
            bar['beat'][j] += ((i-barStart))
        bar = pd.DataFrame(bar)
        generatedNotes = pd.concat([generatedNotes, bar])
        saveFile("save.mid",generatedNotes)
    print("BARS ABOVE THRESHOLD")
    for s in statistics:
        print(s)
    print("TOTAL: "+str(len(statistics) / numBars * 100)+"%")
    return generatedNotes

PITCHES, DURATIONS, BEATS = constSetup()
#PITCHES: ["Key" -> ["10000", "11000" ..2^5-1 total str.. "11111"] "Pitches" -> [32.0, 34.0 ..28(float).. 100.0]]
#DURATIONS: ["Key" -> ["00", "01", "10", "11"] "QuarterValue" -> [0.5, 1, 2, 4 (float)]]
#BEATS: ["Key" -> ["000", "100" .. 2^3 total str.. "111"] "Value" -> [0,1,2,3,4,5,6,7 (int)]]

#Global Parameters
BEAT_NOTES = 3 #Maximum number of notes allowed per beat at the same time
POPULATION_SIZE = 20 #Number of genomes per population

template = "rmelody.mid" #The song to copy

THRESHOLD = 25 #Accepted best score to move on
extracted_template = extract_notes(open_midi(template), str(converter.parse(template).analyze("key")))
BARS = max(extracted_template['bar']) #Automatically calculates number of bars in song
pitchFactor = 1 #How important is the pitch accuracy?

generated_notes = thresholdGenerate(BARS,0,100,False,THRESHOLD,10)

saveFile("output.mid",extracted_template)
