# HINGLISH MAPPING USING METAPHONE ( HMuM )

from pyphonetics import Soundex, Metaphone, RefinedSoundex
from fuzzywuzzy import fuzz
import jellyfish
import eng_to_ipa as phone
import pickle
import nltk


# definition of metaphone_map => a dictionary with key as metaphone code and value as list of words
# unique_words_corpus => a list of all words in all documents

def find_closest_word(close_words_list,word):
    word_fuzz_ratio_dict =  dict()
    for w1 in close_words_list:
        # print(close_words_list)
        # print(w1)
        # word_fuzz_ratio_dict[w1] = fuzz.ratio(phone.convert(w1),phone.convert(word))
        word_fuzz_ratio_dict[w1] = fuzz.ratio(w1,word)
    
    # print(word_fuzz_ratio_dict)

    fuzz_values =  list(word_fuzz_ratio_dict.values())
    max_fuzz =  max(fuzz_values)

    max_fuzz_words = []
    for k,v in word_fuzz_ratio_dict.items():
        if v==max_fuzz:
            max_fuzz_words.append(k)
    
    max_fuzz_words = max_fuzz_words + [word] #adding word to list of closest word
    max_fuzz_words =  list(set(max_fuzz_words))
    max_fuzz_words.sort() #sorting
    
    result_word_index = max_fuzz_words.index(word)
   
    #obtaining index
    if result_word_index == 0:
        result_word_index =1
    elif result_word_index>0:
        result_word_index = result_word_index-1
    
    
    return max_fuzz_words[result_word_index]

def find_closest_word_if_not_in_map(word, metaphone_map):
    code  =  Metaphone().phonetics(word)

    #finding all keys with least fuzz ratio
    
    all_codes =  list(metaphone_map.keys())
    
    code_to_fuzz_map = dict()
    for k in all_codes:
        code_to_fuzz_map[k] =  nltk.edit_distance(code, k)

    fuzz_values =  list(code_to_fuzz_map.values())
    max_fuzz =  min(fuzz_values)
    # print(max_fuzz)

    max_fuzz_codes = []
    for k,v in code_to_fuzz_map.items():
        if v==max_fuzz:
            max_fuzz_codes.append(k)
    
    close_words = list()
    for close_code in max_fuzz_codes:
        close_words = close_words + metaphone_map[close_code]
    
    return find_closest_word(close_words,word)



def map_the_word(word, metaphone_map):
    code =  Metaphone().phonetics(word)

    if code in metaphone_map.keys(): #if metaphone code is present in the map
        close_words_list = metaphone_map[code]
        
        close_word= find_closest_word(close_words_list,word)
        return close_word
    
    elif code not in metaphone_map.keys():
        return find_closest_word_if_not_in_map(word, metaphone_map)
        


def correct_spelling_hinglish(query, unique_words_corpus,metaphone_map) :

    # query = "bulatiiiin haii magur jaaney ka nai" 
    new_query = ""
    word_to_new_word_map = dict()
    for word in query.split():
        
        if word in unique_words_corpus: #if word in corpus no need to map
            word_to_new_word_map[word] = word
            
        else:                           #else map
            new_word = map_the_word(word, metaphone_map)
            word_to_new_word_map[word] = new_word
           

    print(word_to_new_word_map)
    list_of_tokens = query.split()
    new_query = "" + word_to_new_word_map[list_of_tokens[0]]
    for i in range(1,len(list_of_tokens)):
        new_query =  new_query  + " "+word_to_new_word_map[list_of_tokens[i]]
    
    return new_query
