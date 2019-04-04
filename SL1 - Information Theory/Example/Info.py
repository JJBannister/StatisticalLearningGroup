import numpy as np
import math

siddartha_file = "./Data/Siddartha.txt"
metamorphosis_file = "./Data/Metamorphosis.txt"
valid_characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ','.',',','?']

def main():
    sid = _read_txt(siddartha_file)
    meta = _read_txt(metamorphosis_file)

    print("** Siddartha:")
    print(sid[:1000])
    print(".....")
    print()

    print("** Metamorphosis:")
    print(meta[:1000])
    print(".....")
    print()
    input()


    print("*** Maximum Entropy Model ***")
    model = _n_gram(meta, 0) 

    print("Model Shape: ", model.shape)
    print()

    print("Source Entropy: ", _source_entropy(model))
    print()

    print("Sampled Text -- Maximum Entropy Model:")
    print(_generate_text(model))


    input()
    for n in range(1,6):
        print()
        print("*** ", n, " Gram model *** ")
        sid_model = _n_gram(sid, n) 
        meta_model = _n_gram(meta, n) 

        print("Model Shape: ", sid_model.shape)
        print()

        print("Source Entropy -- Siddartha Model: ", _source_entropy(sid_model))
        print("Source Entropy -- Metamorphosis Model: ", _source_entropy(meta_model))
        print()

        print("Sampled Text -- Siddartha Model:")
        print(_generate_text(sid_model))
        print()
        print("Sampled Text -- Metamorphosis Model:")
        print(_generate_text(meta_model))
        print()
        print()
        input()

    _entropy_game(sid)


def _entropy_game(text):
    print("*** The Entropy Game ***")
    print("In each round you will be presented with a block of text. Your task is to gues the next character in the text until you get the correct answer.")
    print("The valid characters are:")
    print(valid_characters)
    input()

    def _entropy(guess_counter):
        if sum(guess_counter) != 0:
            guess_counter = guess_counter/sum(guess_counter)

        def _info(x):
            if x == 0:
                return 0
            return -x * math.log2(x)
        return sum([_info(x) for x in guess_counter])

    n_guesses = np.zeros(30)
    rnd = 0
    while(True):
        print()
        print("*** Round: ", rnd)
        if rnd > 1:
            print("Source Entropy Estimate: ", _entropy(n_guesses))
        print()
        print()
        print()
        rnd += 1

        i = np.random.randint(len(text))
        offset = 500
        next_char = text[i+offset]
        print(text[i:i+offset]+"_")
        print()

        counter = 0
        guess = None
        while(guess != next_char):
            print("What is your guess for the next character?")
            print()
            guess = input()

            if not any([guess == x for x in valid_characters]):
                print("Your guess must be a valid character")
                print("The valid characters are:")
                print(valid_characters)

            else:
                counter += 1
                if guess == next_char:
                    print("Correct! after ", counter, " attempts")
                    print(text[i+offset-50:i+offset+10])
                    n_guesses[counter] += 1
                    break

                elif counter > 5:
                    print("This is just too many guesses... let's try something else.")
                    print("The right answer was '", next_char, "'")
                    print(text[i+offset-50:i+offset+20])
                    n_guesses[counter] += 1
                    break

                else:
                    print("Incorrect... guess again.")
                    print()
                    

def _source_entropy(model):
    n = len(model.shape)

    def _entropy(model):
        def _info(x):
            if x == 0:
                return 0
            return -x * math.log2(x)
        return sum([_info(x) for x in model])

    if n ==1:
        return _entropy(model)

    if n==2:
        ent = 0
        for i in range(len(valid_characters)):
            weight = np.sum(model[i,:])
            if weight != 0:
                dist = model[i,:] / weight
                ent += weight * _entropy(dist)
        return ent

    if n==3:
        ent = 0
        for i in range(len(valid_characters)):
            for j in range(len(valid_characters)):
                weight = np.sum(model[i,j,:])
                if weight != 0:
                    dist = model[i,j,:] / weight
                    ent += weight * _entropy(dist)
        return ent

    if n==4:
        ent = 0
        for i in range(len(valid_characters)):
            for j in range(len(valid_characters)):
                for k in range(len(valid_characters)):
                    weight = np.sum(model[i,j,k,:])
                    if weight != 0:
                        dist = model[i,j,k,:] / weight
                        ent += weight * _entropy(dist)
        return ent

    if n==5:
        ent = 0
        for i in range(len(valid_characters)):
            for j in range(len(valid_characters)):
                for k in range(len(valid_characters)):
                    for l in range(len(valid_characters)):
                        weight = np.sum(model[i,j,k,l,:])
                        if weight != 0:
                            dist = model[i,j,k,l,:] / weight
                            ent += weight * _entropy(dist)
        return ent

def _n_gram(text, n):
    if n==0:
        return np.ones(shape = [len(valid_characters)])/len(valid_characters)

    count = 0
    model = np.zeros(shape = [len(valid_characters)]*n)
    for i in range( len(text)-n-1 ):
        n_gram =[text[i+j] for j in range(n)]
        index = tuple(valid_characters.index(x) for x in n_gram)
        model[index] += 1
        count += 1

    model = model/count
    return model


def _filter_text(text):
    text = [text[i] for i in range(len(text)) if not (text[i] == ' ' and text[i-1] == ' ')]
    text = ''.join([x for x in text if any([x==y for y in valid_characters])])
    return text

def _read_txt(filename):
    f = open(filename, "r")
    lines = f.readlines()
    text = ' '.join(lines).lower()
    f.close()
    return _filter_text(text)


def _generate_text(model):

    n = len(model.shape)
    text = []

    for i in range(1000):
        if n == 1:
            sample_distribution = model

        elif n == 2:
            if i == 0:
                sample_distribution = np.sum(model, axis=(0))
            else:
                sample_distribution = model[text[i-1],:]

        elif n == 3:
            if i == 0:
                sample_distribution = np.sum(model, axis=(0,1))
            elif i == 1:
                marginal_distribution = np.sum(model, axis=(0))
                sample_distribution = marginal_distribution[text[i-1],:]
            else:
                sample_distribution = model[text[i-2],text[i-1],:]

        elif n == 4:
            if i == 0:
                sample_distribution = np.sum(model, axis=(0,1,2))
            elif i == 1:
                marginal_distribution = np.sum(model, axis=(0,1))
                sample_distribution = marginal_distribution[text[i-1],:]
            elif i == 2:
                marginal_distribution = np.sum(model, axis=(0))
                sample_distribution = marginal_distribution[text[i-2],text[i-1],:]
            else:
                sample_distribution = model[text[i-3],text[i-2],text[i-1],:]

        elif n == 5:
            if i == 0:
                sample_distribution = np.sum(model, axis=(0,1,2,3))
            elif i == 1:
                marginal_distribution = np.sum(model, axis=(0,1,2))
                sample_distribution = marginal_distribution[text[i-1],:]
            elif i == 2:
                marginal_distribution = np.sum(model, axis=(0,1))
                sample_distribution = marginal_distribution[text[i-2],text[i-1],:]
            elif i == 3:
                marginal_distribution = np.sum(model, axis=(0))
                sample_distribution = marginal_distribution[text[i-3],text[i-2],text[i-1],:]
            else:
                sample_distribution = model[text[i-4],text[i-3],text[i-2],text[i-1],:]

        else:
            print("6-gram and higher models are not implemented")
            return

        sample_distribution = sample_distribution / np.sum(sample_distribution)
        sample_char = np.random.choice(valid_characters, 1, p=sample_distribution)[0]
        sample_index = valid_characters.index(sample_char)
        text.append(sample_index)

    return ''.join([valid_characters[x] for x in text])

main()
