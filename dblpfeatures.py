"""DBLP Features

Utility class for dblp datasets from version 11.

Requires
--------
json, regex, langid, googletrans, nltk, urlextractor

Classes
-------
DataUtil :
    A class thath can be used to extract some implicit information from a dblp
    dataset (V11+), e.g. the list of co-authors for each publication.
"""

import os
import math
from collections.abc import Iterable
from timeit import default_timer as timer

import json
import regex as re
import langid
from googletrans import Translator
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import urlextract

class DataUtil:
    """
    This class can be used to extract some implicit information from a dblp
    dataset (V11+), e.g. the list of co-authors for each publication. The input
    'jsonData' for most of the methods is expected to be a list-like structure
    where each element is a json object representing a publication (or an author).

    Attributes
    ----------
    authors_dict : dict
        a dictionary where for each <key,value> pair the key is an author's id
        and the value is the author's name
    authors_pubs : dict
        a dictionary where for each <key,value> pair the key is an author's id
        and the value is the list of his publication ids
    publications_dict : dict
        a dictionary where for each <key,value> pair the key is the id of a
        publication and the value is the list of its authors' ids
    titles_list : list
        a list where each entry is the 'cleared' title of a publication; 'cleared'
        means that impurities such as urls, numbers, htmls are removed from the
        original string and then it is stemmed and tokenized
    venues_dict : dict
        a dictionary where for each <key,value> pair the key is a publication
        venue and the value is an ordinal integer associated to the venue
        (it could be used as unique index in a different structure)
    years_dict : dict
        a dictionary where for each <key,value> pair the key is a year of
        publication and the value is an ordinal integer associated to the year
        (it could be used as unique index in a different structure)

    Methods
    -------
    clear_title(sentence:str) -> str
        Removes impurities from the parameter string and applies stemming and
        tokenization to it.

    computeAuthorsInd(auths_keys:Iterable=None) -> None
        Computes a dictionary where each key is an author's id and each value is
        an ordinal integer associated to the author.

    computePublicationsInd() -> None
        Computes a dictionary where each key is the id of a publication and each
        value is an ordinal integer associated to the publication.

    extractAuthors(jsonData:Iterable, save:bool=False, filename:str=None) -> None
        Extracts from the jsonData a dictionary where each key is an author's id
        and each value is the author's name.

    extractCoAuthors(jsonData:Iterable, save:bool=False, filename:str=None) -> None
        Extracts from the jsonData a dictionary where each key is an id of a
        publication and each value is the list of authors' ids of the publication.

    extractPublications(jsonData:Iterable, save:bool=False, filename:str=None) -> None
        Extracts from the jsonData a dictionary where each key is an author's id
        and each value is the list of id of publications published by the author.

    extractTitles(jsonData:Iterable, save:bool=False, filename:str=None) -> (None, list)
        Extracts from the jsonData a list where each entry is a cleared title of
        a publication.

    extractVenues(jsonData:Iterable, save:bool=False, filename:str=None) -> None
        Extracts from the jsonData a dictionary where each key is a publication
        venue and each value is an ordinal integer associated to the venue.

    extractYears(jsonData:Iterable, save:bool=False, filename:str=None) -> None
        Extracts from the jsonData a dictionary where each key is a year of
        publication and each value is an ordinal integer associated to the year.

    loadAttributes(**kwargs) -> None
        Loads the values of the class attributes from files.
    """
    def __init__(self):
        self.authors_dict = None
        self.publications_dict = None
        self.authors_pubs = dict()
        self.titles_list = []
        self.venues_dict = dict()
        self.years_dict = dict()
        self.auth_id_ind = dict()
        self.pub_id_ind = dict()

        self.__lang_predictor = langid.classify
        self.__lang_translator = Translator()
        self.__stop_words = stopwords.words('english')
        self.__url_extractor = urlextract.URLExtract()
        self.__tokenizer = RegexpTokenizer(r'\w+')
        self.__stemmer = PorterStemmer()
        nltk.download('stopwords')

        self.__attr_names = ["authors","coAuthors","publications","titles","venues","years"]

    def clear_title(self, sentence:str) -> str :
        """
        Tokenizes and stems the input title and removes 'impurities' from it;
        'impurities' could be urls, numbers, htmls, etc.

        Parameters:
        ----------
        sentence : str
            a string representing a title

        Returns:
        --------
        str
            a new string obtained tokenizing and stemming the original title
        """
        if(sentence==""):
            return sentence
        # lowercases everything to standardize the sentence
        sentence = sentence.lower()
        # dealing with urls
        urls = set(self.__url_extractor.find_urls(sentence))
        for url in urls:
            sentence = re.sub(url, '', sentence)
        # dealing with numbers
        sentence = re.sub('\d+', '', sentence)
        # dealing with html5 elements (i.e. <br>, <p>, etc.)
        sentence = re.sub('<.*?>', '', sentence)
        # dealing with underscore decorations (i.e. _learning_)
        sentence = re.sub('_+', '', sentence)
        # dealing with hyphen decorations (i.e. -learning-)
        sentence = re.sub('_+', '', sentence)
        # tokenizing
        tokens = self.__tokenizer.tokenize(sentence)
        # dealing with repetitions and exaggerations
        for i in range(len(tokens)):
            tmp = tokens[i]
            for match in re.finditer(r'(.)(\1{1,})', tokens[i]):
                sp = match.span()
                if(sp[1]-sp[0]==2):
                    if(sp[0]==0):
                        tmp = re.sub(match.group(), match.group()[0], tmp)
                    else:
                        if(sp[1]==len(tmp)):
                            tmp = re.sub(match.group(), match.group()[0], tmp)
                else:
                    tmp = re.sub(match.group(), match.group()[0], tmp)
            tokens[i] = tmp
        # stemming
        stemmed = map(lambda token: self.__stemmer.stem(token), tokens)
        tokens = self.__tokenizer.tokenize(" ".join(stemmed))
        # if the created token isn't in the stop words and its length is greater than three, makes it part of "filtered"
        filtered = filter(lambda token: token not in self.__stop_words and len(token) > 3, tokens)
        return " ".join(filtered)

    def computeAuthorsInd(self, auths_keys:Iterable=None) -> None :
        """
        Computes a dictionary where each key is an author's id and each value is
        an ordinal integer associated to the author. The resulting dictionary is
        stored in the class attribute 'self.auth_id_ind'.

        Parameters
        ----------
        auths_keys: Iterable, optional
            if it's not None, it will be used to compute the authors' index,
            otherwise the class attribute "self.authors_dict" is used (default
            is None)

        Returns
        -------
        None
            the resulting dictionary is stored in the class attribute
            'self.auth_id_ind'
        """
        if(self.authors_dict is None and auths_keys is None):
            raise NotImplementedError("can't compute authors indices if authors_dict and auths_keys are both None")
        if(not isinstance(auths_keys,Iterable) and self.authors_dict is None):
            raise TypeError("auths_keys must be an Iterable but %s was passed"%str(type(auths_keys)))

        keys = self.authors_dict if auths_keys is None else auths_keys
        self.auth_id_ind = dict()
        i = 0
        for auth_key in keys:
            self.auth_id_ind[auth_key] = i
            i += 1
        print("Author_id->index completed.")

    def computePublicationsInd(self) -> None :
        """
        Computes a dictionary where each key is the id of a publication and each
        value is an ordinal integer associated to the publication. The resulting
        dictionary is stored in the class attribute 'self.pub_id_ind'.

        Returns
        -------
        None
            the resulting dictionary is stored in the class attribute
            'self.pub_id_ind'
        """
        if(self.publications_dict is None):
            raise NotImplementedError("can't compute publications indices if publications_dict is None")

        self.pub_id_ind = dict()
        i = 0
        for pub in self.publications_dict:
            self.pub_id_ind[pub] = i
            i += 1
        print("Publication_id->index completed.")

    def extractAuthors(self, jsonData:Iterable, save:bool=False, filename:str=None) -> None :
        """
        Extracts from the jsonData a dictionary where each key is an author's id
        and each value is the author's name. The resulting dictionary is stored
        in the class attribute 'self.authors_dict'.

        Parameters
        ----------
        jsonData : Iterable
            a list of json objects representing an author
        save : bool, optional
            if True is passed, then the resulting dictionary is saved on a file
            (default is False)
        filename : str, optional
            if save=True, this filename is used to save the resulting dictionary;
            otherwise its value is ignored (default is None)

        Returns
        -------
        None
            the resulting dictionary is stored in the class attribute
            'self.authors_dict'
        """
        if(not isinstance(jsonData,Iterable)):
            raise TypeError("jsonData must be an Iterable object but %s was passed"%str(type(jsonData)))
        if(len(jsonData)==0):
            raise ValueError("jsonData can't be empty")
        if(save):
            if(not isinstance(filename,str)):
                raise TypeError("path must be a string but %s was passed"%str(type(filename)))
            if(filename==""):
                raise ValueError("filename must contain at least one character")
            ext = os.path.splitext(filename)[1]
            if(ext!=".txt"):
                raise ValueError("the file extension must be \".txt\" but \"%s\" was passed"%ext)

        self.authors_dict = dict()
        i = 0
        start = timer()
        for jsonObj in jsonData:
            self.authors_dict[jsonObj['id']] = jsonObj["name"]
            if(i%10000==0):
                print("\rRows processed: %d/%d"%(i,len(jsonData)),end='',flush=True)
            i += 1
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rRows processed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i,len(jsonData),mins,secs))
        if(save):
            with open(filename, 'w') as file:
                json.dump(self.authors_dict, file)

    def extractCoAuthors(self, jsonData:Iterable, save:bool=False, filename:str=None) -> None :
        """
        Extracts from the jsonData a dictionary where each key is an id of a
        publication and each value is the list of authors' ids of the publication.
        The resulting dictionary is stored in the class attribute
        'self.publications_dict'.

        Parameters
        ----------
        jsonData : Iterable
            a list of json objects representing a publication
        save : bool, optional
            if True is passed, then the resulting dictionary is saved on a file
            (default is False)
        filename : str, optional
            if save=True, this filename is used to save the resulting dictionary;
            otherwise its value is ignored (default is None)

        Returns
        -------
        None
            the resulting dictionary is stored in the class attribute
            'self.publications_dict'
        """
        if(not isinstance(jsonData,Iterable)):
            raise TypeError("jsonData must be an Iterable object but %s was passed"%str(type(jsonData)))
        if(len(jsonData)==0):
            raise ValueError("jsonData can't be empty")
        if(save):
            if(not isinstance(filename,str)):
                raise TypeError("path must be a string but %s was passed"%str(type(filename)))
            if(filename==""):
                raise ValueError("filename must contain at least one character")
            ext = os.path.splitext(filename)[1]
            if(ext!=".txt"):
                raise ValueError("the file extension must be \".txt\" but \"%s\" was passed"%ext)

        self.publications_dict = dict()
        i = 0
        start = timer()
        for jsonObj in jsonData:
            coauthors = []
            for author in jsonObj['authors']:
                coauthors.append(author['id'])
            self.publications_dict[jsonObj['id']] = list(set(coauthors))
            if(i%10000==0):
                print("\rRows processed: %d/%d"%(i,len(jsonData)),end='',flush=True)
            i += 1
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rRows processed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i,len(jsonData),mins,secs))
        if(save):
            with open(filename, 'w') as file:
                json.dump(self.publications_dict, file)

    def extractPublications(self, jsonData:Iterable, save:bool=False, filename:str=None) -> None :
        """
        Extracts from the jsonData a dictionary where each key is an author's id
        and each value is the list of id of publications published by the author.
        The resulting dictionary is stored in the class attribute
        'self.authors_pubs'.

        Parameters
        ----------
        jsonData : Iterable
            a list of json objects representing an author
        save : bool, optional
            if True is passed, then the resulting dictionary is saved on a file
            (default is False)
        filename : str, optional
            if save=True, this filename is used to save the resulting dictionary;
            otherwise its value is ignored (default is None)

        Returns
        -------
        None
            the resulting dictionary is stored in the class attribute
            'self.authors_pubs'
        """
        if(not isinstance(jsonData,Iterable)):
            raise TypeError("jsonData must be an Iterable object but %s was passed"%str(type(jsonData)))
        if(len(jsonData)==0):
            raise ValueError("jsonData can't be empty")
        if(save):
            if(not isinstance(filename,str)):
                raise TypeError("path must be a string but %s was passed"%str(type(filename)))
            if(filename==""):
                raise ValueError("filename must contain at least one character")
            ext = os.path.splitext(filename)[1]
            if(ext!=".txt"):
                raise ValueError("the file extension must be \".txt\" but \"%s\" was passed"%ext)

        self.authors_pubs = dict()
        i = 0
        for jsonObj in jsonData:
            pubs = []
            for pub in jsonObj['pubs']:
                pubs.append(pub['id'])
            self.authors_pubs[jsonObj['id']] = pubs
            if(i%10000==0):
                print("\rComputed: %d/%d"%(i,len(jsonData)),end='',flush=True)
            i += 1
        print("\rComputed: %d/%d"%(i,len(jsonData)))
        if(save):
            with open(filename, 'w') as file:
                json.dump(self.authors_pubs, file)

    def extractTitles(self, jsonData:Iterable, save:bool=False, filename:str=None) -> (None,list) :
        """
        Extracts from the jsonData a list where each entry is a cleared title of
        a publication (see 'clear_title() method'). The resulting list is stored
        in the class attribute 'self.titles_list'. During the computation it is
        required to have a stable internet connection due to usage of googletrans
        library; anyway, some errors could occurr: in this case, the faulty
        titles are added to a separate list (which is returned) and in the main
        list is added a symbolic string "+" (which can be used to find where the
        error occurred).

        Parameters
        ----------
        jsonData : Iterable
            a list of json objects representing a publication
        save : bool, optional
            if True is passed, then the resulting list is saved on a file
            (default is False)
        filename : str, optional
            if save=True, this filename is used to save the resulting list;
            otherwise its value is ignored (default is None)

        Returns
        -------
        None or list
            the resulting list is stored in the class attribute
            'self.titles_list'; if some errors occurr, a list with the faulty
            titles is returned
        """
        if(not isinstance(jsonData,Iterable)):
            raise TypeError("jsonData must be an Iterable object but %s was passed"%str(type(jsonData)))
        if(len(jsonData)==0):
            raise ValueError("jsonData can't be empty")
        if(save):
            if(not isinstance(filename,str)):
                raise TypeError("path must be a string but %s was passed"%str(type(filename)))
            if(filename==""):
                raise ValueError("filename must contain at least one character")
            ext = os.path.splitext(filename)[1]
            if(ext!=".txt"):
                raise ValueError("the file extension must be \".txt\" but \"%s\" was passed"%ext)

        self.titles_list = []
        specials = []
        i = 0
        start = timer()
        for jsonObj in jsonData:
            title = jsonObj['title']
            try:
                if(self.__lang_predictor(title)[0]!='en'):
                    title = self.__lang_translator.translate(title).text
                self.titles_list.append(self.clear_title(title))
            except:
                specials.append(title)
                self.titles_list.append("+")
            if(i%100==0):
                print("\rRows processed: %d/%d"%(i,len(jsonData)),end='',flush=True)
            i += 1
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rRows processed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i,len(jsonData),mins,secs))
        if(save):
            with open(filename, 'w', encoding='utf-8') as file:
                for i in range(len(self.titles_list)):
                    file.write("%s%s"%(self.titles_list[i],("" if i==len(self.titles_list)-1 else "\n")))
                    if(i%10000==0):
                        print("\rLines written: %d/%d"%(i,len(self.titles_list)),end='',flush=True)
                print("\rLines written: %d/%d"%(i+1,len(self.titles_list)))
        if(len(specials)>0):
            return specials

    def extractVenues(self, jsonData:Iterable, save:bool=False, filename:str=None) -> None :
        """
        Extracts from the jsonData a dictionary where each key is a publication
        venue and each value is an ordinal integer associated to the venue. The
        resulting dictionary is stored in the class attribute 'self.venues_dict'.

        Parameters
        ----------
        jsonData : Iterable
            a list of json objects representing a publication
        save : bool, optional
            if True is passed, then the resulting dictionary is saved on a file
            (default is False)
        filename : str, optional
            if save=True, this filename is used to save the resulting dictionary;
            otherwise its value is ignored (default is None)

        Returns
        -------
        None
            the resulting dictionary is stored in the class attribute
            'self.venues_dict'
        """
        if(not isinstance(jsonData,Iterable)):
            raise TypeError("jsonData must be an Iterable object but %s was passed"%str(type(jsonData)))
        if(len(jsonData)==0):
            raise ValueError("jsonData can't be empty")
        if(save):
            if(not isinstance(filename,str)):
                raise TypeError("path must be a string but %s was passed"%str(type(filename)))
            if(filename==""):
                raise ValueError("filename must contain at least one character")
            ext = os.path.splitext(filename)[1]
            if(ext!=".txt"):
                raise ValueError("the file extension must be \".txt\" but \"%s\" was passed"%ext)

        venues = []
        i = 0
        start = timer()
        for jsonObj in jsonData:
            try:
                venues.append(jsonObj['venue']['raw'])
            except:
                pass
            if(i%10000==0):
                print("\rRows processed: %d/%d"%(i,len(jsonData)),end='',flush=True)
            i += 1
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rRows processed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i,len(jsonData),mins,secs))
        venues = list(set(venues))
        self.venues_dict = { str(venues[i]) : i for i in range(len(venues)) }
        if(save):
            with open(filename, 'w') as file:
                json.dump(self.venues_dict, file)

    def extractYears(self, jsonData:Iterable, save:bool=False, filename:str=None) -> None :
        """
        Extracts from the jsonData a dictionary where each key is a year of
        publication and each value is an ordinal integer associated to the year.
        The resulting dictionary is stored in the class attribute
        'self.years_dict'.

        Parameters
        ----------
        jsonData : Iterable
            a list of json objects representing a publication
        save : bool, optional
            if True is passed, then the resulting dictionary is saved on a file
            (default is False)
        filename : str, optional
            if save=True, this filename is used to save the resulting dictionary;
            otherwise its value is ignored (default is None)

        Returns
        -------
        None
            the resulting dictionary is stored in the class attribute
            'self.years_dict'
        """
        if(not isinstance(jsonData,Iterable)):
            raise TypeError("jsonData must be an Iterable object but %s was passed"%str(type(jsonData)))
        if(len(jsonData)==0):
            raise ValueError("jsonData can't be empty")
        if(save):
            if(not isinstance(filename,str)):
                raise TypeError("path must be a string but %s was passed"%str(type(filename)))
            if(filename==""):
                raise ValueError("filename must contain at least one character")
            ext = os.path.splitext(filename)[1]
            if(ext!=".txt"):
                raise ValueError("the file extension must be \".txt\" but \"%s\" was passed"%ext)

        years = set()
        i = 0
        start = timer()
        for jsonObj in jsonData:
            if('year' in jsonObj):
                years = years|set([jsonObj['year']])
            if(i%10000==0):
                print("\rRows processed: %d/%d"%(i,len(jsonData)),end='',flush=True)
            i += 1
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rRows processed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i,len(jsonData),mins,secs))
        years = list(years)
        self.years_dict = { str(years[i]) : i for i in range(len(years)) }
        if(save):
            with open(filename, 'w') as file:
                json.dump(self.years_dict, file)

    def loadAttributes(self, **kwargs) -> None :
        """
        Loads the values of the class attributes from files. In order to load
        a specific attribute value from a file, it's required to pass a kwarg
        with value set to the path of the file which contains the attribute
        values (e.g. venues="./venues.txt" to load the attribute self.venues_dict).

        Parameters
        ----------
        **kwargs :
            one (or more) from { authors,coAuthors,publications,titles,venues,
            years }; other kwargs will be simply ignored without raising any
            error

        Returns
        -------
        None
            the values loaded with valid kwargs will be stored in the class
            attributes
        """
        if(len(kwargs)==0):
            return
        for attr in kwargs:
            if(attr in self.__attr_names):
                path = kwargs[attr]
                if(attr=="authors"):
                    with open(path, 'r') as file:
                        self.authors_dict = json.load(file)
                elif(attr=="coAuthors"):
                    with open(path, 'r') as file:
                        self.publications_dict = json.load(file)
                elif(attr=="publications"):
                    with open(path, 'r') as file:
                        self.authors_pubs = json.load(file)
                elif(attr=="titles"):
                    with open(path, 'r', encoding='utf-8') as file:
                        self.titles_list = [line.rstrip() for line in file]
                elif(attr=="venues"):
                    with open(path, 'r') as file:
                        self.venues_dict = json.load(file)
                else: #attr=="years"
                    with open(path, 'r') as file:
                        self.years_dict = json.load(file)
                print("%s attribute loaded"%attr)
