"""DBLP Utils

Utility class for dblp datasets from version 11.

Requires
--------
json, pandas, numpy, regex

Classes
-------
IOUtil :
    A class that can be used to load and store the dblp datasets (V11+) in various
    representations.

CitationDataset :
    A class that can be used to dynamically load a dblp dataset (V11+) from a file
    on the disk.
"""

import os
import math
from collections.abc import Iterable
from timeit import default_timer as timer

import json
from json import JSONDecodeError
import pandas as pd
pandas = pd
import numpy as np
import regex as re

class IOUtil:
    """
    This class can be used to load and store the dblp datasets (V11+) in various
    representations. The supported representations are list of json entries
    and pandas.DataFrame.

    Methods
    -------
    copyLinesFromFile(path:str, lines:int, filename:str) -> None
        Stores a number of lines from a file to a new file.

    dumpLinesFromJson(jsonData:Iterable, filename:str, keys:Iterable=None) -> None
        Stores a new file containing a line for each entry in the jsonData
        parameter.

    extractAuthorsDataset(jsonData:Iterable, auth_keys:Iterable=None, pub_keys:Iterable=None, pub_nested:dict=None) -> dict
        Extracts the dataset of authors from the publications dataset and stores
        it in a dictionary.

    jsonToDataFrame(jsonData:Iterable, keys:Iterable) -> pandas.DataFrame
        Creates a DataFrame object from the jsonData parameter.

    loadAsDataframe(path:str, keys:Iterable, max_lines:int=-1,file_lines:int=-1) -> pandas.DataFrame
        Loads the data from the disk and returns a DataFrame representation of
        each line.

    loadAsJson(path:str, max_lines:int=-1) -> list
        Loads the data from the disk and returns it as a list of json objects.

    loadKeysLists(path:str, separator:str, enc:str='cp1252') -> dict
        Loads some data from the disk and stores it in a dictionary.

    loadKeysVals(path:str, separator:str, enc:str='utf-8') -> dict
        Loads some data from the disk and stores it in a dictionary.

    selectRandom(data, num_random:int, filename:str) -> None
        Selects a number of random entries in data and stores them in a new
        file.

    selectRandomFromFile(path:str, num_random:int, filename:str, data_len:int=-1) -> None
        Selects a number of random lines from the source file and stores
        them in a new file.

    selectRandomPro(data, num_random:int, filename:str) -> None
        Selects a number of random entries in data, considering also the cited
        entries in the already selected ones, and stores them in a new file.
    """

    def copyLinesFromFile(self, path:str, lines:int, filename:str) -> None :
        """
        Copies a number of lines equal to 'lines' from the source file (path) and
        stores them in the destination file (filename).

        Parameters
        ----------
        path : str
            the path to the source file
        lines : int
            the number of lines that have to be copied
        filename : str
            the path of the destination file

        Returns
        -------
        None
        """
        if(not isinstance(path,str)):
            raise TypeError("path must be a string but %s was passed"%str(type(path)))
        if(not isinstance(filename,str)):
            raise TypeError("path must be a string but %s was passed"%str(type(filename)))
        if(not isinstance(lines,int)):
            raise TypeError("lines must be an int but %s was passed"%str(type(lines)))
        if(not os.path.exists(path) or not os.path.isfile(path)):
            raise ValueError("invalid path to file")
        if(lines<=0):
            raise ValueError("lines must be a positive number greater than 0 but %d was passed"%lines)
        if(filename==""):
            raise ValueError("filename must contain at least one character")
        ext = os.path.splitext(filename)[1]
        if(ext!=".txt"):
            raise ValueError("the file extension must be \".txt\" but \"%s\" was passed"%ext)

        with open(path, 'r') as in_file, open(filename, 'w') as out_file:
            start = timer()
            i = 0
            line = in_file.readline()
            while(i<lines and line!=""):
                out_file.write(line)
                if(i%10000==0):
                    print("\rLines written: %d/%d"%(i,lines),end='',flush=True)
                i += 1
                line = in_file.readline()
            end = timer()
            elapsed = round(end-start,2)
            mins = math.floor(elapsed/60)
            secs = math.ceil(elapsed-mins*60)
            print("\rLines written: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i,lines,mins,secs))

    def dumpLinesFromJson(self, jsonData:Iterable, filename:str, keys:Iterable=None) -> None :
        """
        Stores a new file containing a line for each entry in the jsonData parameter.
        Each line is stored in its original json format.

        Parameters
        ----------
        jsonData : Iterable
            a list of json objects
        filename : str
            the path of the destination file
        keys : Iterable, optional
            the keys of the json objects that have to be saved (default is None and
            indicates that each key must be saved)

        Returns
        -------
        None
        """
        if(not isinstance(jsonData,Iterable)):
            raise TypeError("jsonData must be an Iterable object but %s was passed"%str(type(jsonData)))
        if(not isinstance(filename,str)):
            raise TypeError("path must be a string but %s was passed"%str(type(filename)))
        if(keys is not None and not isinstance(keys,Iterable)):
            raise TypeError("keys must be an Iterable object but %s was passed"%str(type(keys)))
        if(len(jsonData)==0):
            return
        if(filename==""):
            raise ValueError("filename must contain at least one character")
        ext = os.path.splitext(filename)[1]
        if(ext!=".txt"):
            raise ValueError("the file extension must be \".txt\" but \"%s\" was passed"%ext)

        with open(filename, 'w') as out_file:
            start = timer()
            i = 0
            for entry in jsonData:
                if(keys is None or len(keys)==0):
                    out_file.write("%s%s"%(json.dumps(entry),("" if i==len(jsonData)-1 else "\n")))
                else:
                    obj = dict()
                    for key in keys:
                        try:
                            obj[key] = entry[key]
                        except KeyError:
                            obj[key] = ""
                    out_file.write("%s%s"%(json.dumps(obj),("" if i==len(jsonData)-1 else "\n")))
                if(i%10000==0):
                    print("\rLines written: %d/%d"%(i,len(jsonData)),end='',flush=True)
                i += 1
            end = timer()
            elapsed = round(end-start,2)
            mins = math.floor(elapsed/60)
            secs = math.ceil(elapsed-mins*60)
            print("\rLines written: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i,len(jsonData),mins,secs))

    def extractAuthorsDataset(self, jsonData:Iterable, auth_keys:Iterable=None, pub_keys:Iterable=None, pub_nested:dict=None) -> dict :
        """
        Extracts the dataset of authors from the publications dataset and stores
        it in a dictionary. Each entry in the dictionary has the author id as
        key and a json representation as value (similar to the representation of
        a publication); the id of the author is also stored in the json value if
        specified. A general entry contains at least the keys "id" and "pubs"
        and a general publication entry in "pubs" contains at least "id" and
        "references".

        Parameters
        ----------
        jsonData : Iterable
            a list of json objects representing a publication
        auth_keys : Iterable, optional
            the keys relative to an author that have to be included in the json
            representation, if specified
        pub_keys : Iterable, optional
            the keys relative to a publication that have to be included in the
            json representing the publication of an author, if specified
        pub_nested : dict, optional
            a dictionary representing primary key and nested key of a key that
            have to be included in the json representing the publication of an
            author, if specified; in the end, only the primary key will be
            included (e.g. for {"fos":"name"} will be searched for
            json["fos"]["name"] and this value will be saved under json["fos"])

        Returns
        -------
        dict
            a dictionary which contains the authors dataset with this structure:
            dict := {<author_id>:<author_info>}
                <author_id> := str
                <author_info> := {"id":str, auth_keys, "pubs":list<pub_info>}
                    <pub_info> := {"id":str, pub_keys, pub_nested,
                                   "references":list<str>}
        """
        if(not isinstance(jsonData,Iterable)):
            raise TypeError("jsonData must be an Iterable object but %s was passed"%str(type(jsonData)))
        if(len(jsonData)==0):
            return None
        if(auth_keys is not None and not isinstance(auth_keys,Iterable)):
            raise TypeError("auth_keys must be an Iterable object but %s was passed"%str(type(auth_keys)))
        if(pub_keys is not None and not isinstance(pub_keys,Iterable)):
            raise TypeError("pub_keys must be an Iterable object but %s was passed"%str(type(pub_keys)))
        if(pub_nested is not None and not isinstance(pub_nested,dict)):
            raise TypeError("pub_nested must be an Iterable object but %s was passed"%str(type(pub_nested)))

        dataset = dict()
        i = 0
        start = timer()
        for jsonObj in jsonData:
            pub = {"id":jsonObj['id']}
            for key in pub_keys:
                if(key in jsonObj):
                    pub[key] = jsonObj[key]
                    if(isinstance(pub[key],str) and len(re.sub("\s+", '', pub[key]))==0):
                        pub[key] = ""
                else:
                    pub[key] = ""
            for key in pub_nested:
                try:
                    pub[key] = jsonObj[key][pub_nested[key]]
                    if(isinstance(pub[key],str) and len(re.sub("\s+", '', pub[key]))==0):
                        pub[key] = ""
                except:
                    pub[key] = ""
            if('references' in jsonObj):
                pub['references'] = jsonObj['references']
            else:
                pub['references'] = ""
            for author in jsonObj['authors']:
                idx = author['id']
                if(not idx in dataset):
                    obj = {"id":idx}
                    for key in auth_keys:
                        if(key in author):
                            obj[key] = author[key]
                            if(isinstance(obj[key],str) and len(re.sub("\s+", '', obj[key]))==0):
                                obj[key] = ""
                        else:
                            obj[key] = ""
                    obj["pubs"] = [pub]
                    dataset[idx] = obj
                else:
                    dataset[idx]["pubs"].append(pub)
            if(i%10000==0):
                print("\rRows processed: %d/%d"%(i,len(jsonData)),end='',flush=True)
            i += 1
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rRows processed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i,len(jsonData),mins,secs))
        return dataset

    def jsonToDataframe(self, jsonData:Iterable, keys:Iterable) -> pandas.DataFrame :
        """
        Creates a DataFrame object from the jsonData parameter.

        Parameters
        ----------
        jsonData : Iterable
            a list of json objects
        keys : Iterable
            the keys that can be accessed in each json object; they will be used
            as column names for the DataFrame

        Returns
        -------
        pandas.DataFrame
            a DataFrame object where the i-th row corresponds to the i-th json
            object in jsonData
        """
        if(not isinstance(jsonData,Iterable)):
            raise TypeError("jsonData must be an Iterable object but %s was passed"%str(type(jsonData)))
        if(not isinstance(keys,Iterable)):
            raise TypeError("keys must be an Iterable object but %s was passed"%str(type(keys)))
        if(len(keys)==0):
            raise ValueError("keys can't be empty")

        n_rows = len(jsonData)
        if(n_rows==0):
            return pd.DataFrame(columns=keys)
        n_cols = len(keys)
        data = [None]*n_rows
        print("Reading jsonData...")
        for r in range(n_rows):
            row = [None]*n_cols
            for c in range(n_cols):
                try:
                    row[c] = jsonData[r][keys[c]]
                except KeyError:
                    pass
            data[r] = row
        print("Returning DataFrame...")
        return pd.DataFrame(data, columns=keys)

    def loadAsDataframe(self, path:str, keys:Iterable, max_lines:int=-1, file_lines:int=-1) -> pandas.DataFrame :
        """
        Loads the dataset from the disk. The method assumes that
        the data is stored in a file where each line is a json string.

        Parameters
        ----------
        path : str
            the path to the data on the disk
        keys : Iterable
            the keys that can be accessed in each json object; they will be used
            as column names for the DataFrame
        max_lines : int, optional
            the maximum number of lines of the file that has to be read (default
            is -1 and indicates all the lines)
        file_lines : int, optional
            the total number of lines written on the file (default is -1 and
            indicates that this quantity is unknown)

        Returns
        -------
        pandas.DataFrame
            a DataFrame object where each row is a line written on the file
        """
        if(not isinstance(path,str)):
            raise TypeError("path must be a string but %s was passed"%str(type(path)))
        if(not isinstance(keys,Iterable)):
            raise TypeError("keys must be an Iterable object but %s was passed"%str(type(keys)))
        if(not isinstance(max_lines,int)):
            raise TypeError("max_lines must be an int but %s was passed"%str(type(max_lines)))
        if(not isinstance(file_lines,int)):
            raise TypeError("file_lines must be an int but %s was passed"%str(type(file_lines)))
        if(not os.path.exists(path) or not os.path.isfile(path)):
            raise ValueError("invalid path to file")
        if(len(keys)==0):
            raise ValueError("keys can't be empty")
        if(max_lines<=0 and max_lines!=-1):
            raise ValueError("max_lines must be a positive number greater than 0 but %d was passed"%max_lines)
        if(file_lines<=0 and file_lines!=-1):
            file_lines = -1

        data = None
        if(file_lines==-1):
            print("Counting lines...")
            file_lines = sum(1 for line in open(path, 'r'))
        if(max_lines!=-1):
            max_lines = min(max_lines, file_lines)
        else:
            max_lines = file_lines
        with open(path, 'r') as file:
            print("Reading lines...")
            start = timer()
            data = [None]*max_lines
            line_num = 0
            line = file.readline()
            while(line_num<max_lines):
                data[line_num] = json.loads(line.rstrip())
                row = [None]*len(keys)
                for c in range(len(keys)):
                    try:
                        row[c] = data[line_num][keys[c]]
                    except KeyError:
                        pass
                data[line_num] = row
                if(line_num%10000==0):
                    print("\rLines read: %d/%d"%(line_num,max_lines),end='',flush=True)
                line_num += 1
                line = file.readline()
            end = timer()
            elapsed = round(end-start,2)
            mins = math.floor(elapsed/60)
            secs = math.ceil(elapsed-mins*60)
            print("\rLines read: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(line_num,max_lines,mins,secs))
        print("Returning DataFrame...")
        return pd.DataFrame(data, columns=keys)

    def loadAsJson(self, path:str, max_lines:int=-1) -> list :
        """
        Loads the dataset from the disk. The method assumes that
        the data is stored in a file where each line is a json string.

        Parameters
        ----------
        path : str
            the path to the data on the disk
        max_lines : int, optional
            the maximum number of lines of the file that has to be read (default
            is -1 and indicates all the lines)

        Returns
        -------
        list
            a list of json object
        """
        if(not isinstance(path,str)):
            raise TypeError("path must be a string but %s was passed"%str(type(path)))
        if(not isinstance(max_lines,int)):
            raise TypeError("max_lines must be an int but %s was passed"%str(type(max_lines)))
        if(not os.path.exists(path) or not os.path.isfile(path)):
            raise ValueError("invalid path to file")
        if(max_lines<=0 and max_lines!=-1):
            raise ValueError("max_lines must be a positive number greater than 0 but %d was passed"%max_lines)

        data = None
        with open(path, 'r') as file:
            print("Reading lines...")
            start = timer()
            if(max_lines==-1):
                data = [json.loads(line.rstrip()) for line in file]
            else:
                data = [None]*max_lines
                line_num = 0
                line = file.readline()
                while(line_num<max_lines and line!=""):
                    data[line_num] = json.loads(line.rstrip())
                    line_num += 1
                    line = file.readline()
                if(line_num==0):
                    del data
                    data = None
                if(line==""):
                    del data[line_num:]
            end = timer()
            elapsed = round(end-start,2)
            mins = math.floor(elapsed/60)
            secs = math.ceil(elapsed-mins*60)
            print("\rLines read: %d\t| Elapsed: %d min(s) %d sec(s)"%(len(data),mins,secs))
        return data

    def loadKeysLists(self, path:str, separator:str, enc:str='cp1252') -> dict :
        """
        Loads some data from a file and stores it in a dictionary.
        The data is expected to be formatted as <key><separator><value>,
        where value is a list (comma separated values). The resulting value
        is returned as a string; the behaviour of this method with complex
        values is unknown.

        Parameters
        ----------
        path : str
            the path to the data on the disk
        separator : str
            the separator used to separate key-value pairs
        enc : str, optional
            a string representing the encoding

        Returns
        -------
        dict
            a simple dictionary
        """
        if(not isinstance(path,str)):
            raise TypeError("path must be a string but %s was passed"%str(type(path)))
        if(not isinstance(separator, str)):
            raise TypeError("separator must be of type str but %s was passed"%str(type(separator)))
        if(not os.path.exists(path) or not os.path.isfile(path)):
            raise ValueError("invalid path to file")
        if(separator==""):
            raise ValueError("separator must contain at least one character")
        if(not isinstance(enc,str)):
            enc='cp1252'

        dictionary = dict()
        start = timer()
        with open(path, 'r', encoding=enc) as file:
            i = 0
            line = file.readline()
            while(line!=""):
                line = line.split(separator)
                line[1] = line[1].strip("][\n")
                if(line[1]==''):
                    dictionary[line[0]] = []
                else:
                    dictionary[line[0]] = [tk.strip("''") for tk in line[1].split(", ")]
                if(i%10000==0):
                    print("\rRows processed: %d"%i,end='',flush=True)
                i += 1
                line = file.readline()
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rRows processed: %d\t| Elapsed: %d min(s) %d sec(s)"%(i,mins,secs))
        return dictionary

    def loadKeysVals(self, path:str, separator:str, enc:str='utf-8') -> dict :
        """
        Loads some data from a file and stores it in a dictionary.
        The data is expected to be formatted as <key><separator><value>,
        where value is a primitive object. The behaviour of this method with
        complex values is unknown.

        Parameters
        ----------
        path : str
            the path to the data on the disk
        separator : str
            the separator used to separate key-value pairs
        enc : str, optional
            a string representing the encoding

        Returns
        -------
        dict
            a simple dictionary
        """
        if(not isinstance(path,str)):
            raise TypeError("path must be a string but %s was passed"%str(type(path)))
        if(not isinstance(separator,str)):
            raise TypeError("separator must be a string but %s was passed"%str(type(separator)))
        if(not os.path.exists(path) or not os.path.isfile(path)):
            raise ValueError("invalid path to file")
        if(separator==""):
            raise ValueError("separator must contain at least one character")
        if(not isinstance(enc,str)):
            enc='utf-8'

        dictionary = dict()
        start = timer()
        with open(path, 'r', encoding=enc) as file:
            i = 0
            line = file.readline()
            while(line!=""):
                line = line.split(separator)
                dictionary[line[0]] = line[1].split("\n")[0]
                if(i%10000==0):
                    print("\rRows processed: %d"%i,end='',flush=True)
                i += 1
                line = file.readline()
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rRows processed: %d\t| Elapsed: %d min(s) %d sec(s)"%(i,mins,secs))
        return dictionary

    def selectRandom(self, data, num_random:int, filename:str) -> None :
        """
        Selects 'num_random' random entries from the data and stores them
        in a destination file.

        Parameters
        ----------
        data :
            an istance of numpy.ndarray or numpy.generic containing the data
            that have to be saved
        num_random : int
            the number of random entries that have to be selected
        filename : str
            the path of the destination file

        Returns
        -------
        None
        """
        if(not isinstance(data, (np.ndarray, np.generic))):
            raise TypeError("data must be of type Iterable but %s was passed"%str(type(data)))
        if(not isinstance(num_random, int)):
            raise TypeError("num_random must be of type int but %s was passed"%str(type(num_random)))
        if(not isinstance(filename, str)):
            raise TypeError("filename must be of type str but %s was passed"%str(type(filename)))
        if(filename==""):
            raise ValueError("filename must contain at least one character")
        ext = os.path.splitext(filename)[1]
        if(ext!=".txt"):
            raise ValueError("the file extension must be \".txt\" but \"%s\" was passed"%ext)

        if(num_random==len(data)):
            self.dumpLinesFromJson(data, filename)
        else:
            indices = sorted(np.random.choice(len(data), num_random, replace=False))
            random_selected = data[indices]
            self.dumpLinesFromJson(random_selected, filename)

    def selectRandomFromFile(self, path:str, num_random:int, filename:str, file_lines:int=-1) -> None :
        """
        Selects 'num_random' random lines from the source file (path) and
        stores them in a destination file (filename).

        Parameters
        ----------
        path :
            the path to the data on the disk
        num_random : int
            the number of random lines that have to be selected
        filename : str
            the path of the destination file
        file_lines : int, optional
            the total number of lines written on the source file (default is -1
            and indicates that this quantity is unknown)

        Returns
        -------
        None
        """
        if(not isinstance(path, str)):
            raise TypeError("path must be of type str but %s was passed"%str(type(path)))
        if(not isinstance(filename, str)):
            raise TypeError("filename must be of type str but %s was passed"%str(type(filename)))
        if(not isinstance(num_random, int)):
            raise TypeError("num_random must be of type int but %s was passed"%str(type(num_random)))
        if(not isinstance(file_lines,int)):
            raise TypeError("file_lines must be an int but %s was passed"%str(type(file_lines)))
        if(not os.path.exists(path) or not os.path.isfile(path)):
            raise ValueError("invalid path to file")
        if(filename==""):
            raise ValueError("filename must contain at least one character")
        ext = os.path.splitext(filename)[1]
        if(ext!=".txt"):
            raise ValueError("the file extension must be \".txt\" but \"%s\" was passed"%ext)
        if(file_lines<=0 and file_lines!=-1):
            file_lines = -1

        if(file_lines==-1):
            print("Counting lines...")
            file_lines = sum(1 for line in open(path, 'r'))
        indices = sorted(np.random.choice(file_lines, num_random, replace=False))
        with open(path, 'r') as in_file, open(filename, 'w') as out_file:
            start = timer()
            i = 0
            copied = 0
            next_index = indices[copied]
            line = in_file.readline()
            while(i<file_lines or copied<num_random):
                if(i==next_index):
                    out_file.write(line)
                    copied += 1
                    if(copied<num_random):
                        next_index = indices[copied]
                if(copied%10000==0):
                    print("\rLines copied: %d/%d"%(copied,num_random),end='',flush=True)
                i += 1
                line = in_file.readline()
            end = timer()
            elapsed = round(end-start,2)
            mins = math.floor(elapsed/60)
            secs = math.ceil(elapsed-mins*60)
            print("\rLines copied: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(copied,num_random,mins,secs))

    def selectRandomPro(self, data, num_random:int, filename:str) -> None :
        """
        Selects at least 'num_random' random entries from the data and stores
        them in a destination file. In this version, the publications cited in
        the selected ones are also included; for this reason the final number of
        selected publications could greater than the specified parameter.

        Parameters
        ----------
        data :
            an istance of numpy.ndarray or numpy.generic containing the data
            that have to be saved (jsons representing publications)
        num_random : int
            the number of random entries that have to be selected
        filename : str
            the path of the destination file

        Returns
        -------
        None
        """
        if(not isinstance(data, (np.ndarray, np.generic))):
            raise TypeError("data must be of type Iterable but %s was passed"%str(type(data)))
        if(not isinstance(num_random, int)):
            raise TypeError("num_random must be of type int but %s was passed"%str(type(num_random)))
        if(not isinstance(filename, str)):
            raise TypeError("filename must be of type str but %s was passed"%str(type(filename)))
        if(filename==""):
            raise ValueError("filename must contain at least one character")
        ext = os.path.splitext(filename)[1]
        if(ext!=".txt"):
            raise ValueError("the file extension must be \".txt\" but \"%s\" was passed"%ext)

        if(num_random==len(data)):
            self.dumpLinesFromJson(data, filename)
        else:
            tmp = dict()
            for pub in data:
                tmp[pub['id']] = pub
            selected = dict()
            while(len(selected)<num_random):
                ind = np.random.randint(len(data))
                while(data[ind]['id'] in selected):
                    ind = np.random.randint(len(data))
                random_selected = data[ind]
                selected[random_selected['id']] = random_selected
                flag = True
                while(flag):
                    flag = False
                    selections = list(selected.keys()).copy()
                    for selection in selections:
                        for ref_id in selected[selection]['references']:
                            if(ref_id not in selected):
                                selected[ref_id] = tmp[ref_id]
                                flag = True
                if(len(selected)%10000==0):
                    print("\rRandomly selected: %d/%d"%(len(selected),num_random),end='',flush=True)
            print("\rRandomly selected: %d/%d"%(len(selected),num_random))
            self.dumpLinesFromJson(selected.values(), filename)

class CitationDataset:
    """
    This class can be used to dynamically load a dblp dataset (V11+) from a file
    on the disk. It can be used to save space in memory if the dataset is too big.
    """
    def __init__(self, path:str, length:int=None):
        if(not isinstance(path,str)):
            raise TypeError("path must be of type str but %s was passed"%str(type(path)))
        if(not os.path.exists(path) or not os.path.isfile(path)):
            raise ValueError("invalid path to file")
        if(length!=None and length<0):
            length=None
        self.length = sum(1 for line in open(path, 'r')) if length==None else length
        self.path = path

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.__CiteIterator(self)

    class __CiteIterator:
        # This inner class loads a line per '__next__' invocation, saving space
        # in memory.
        def __init__(self, ds):
            self.__ds = ds
            self.__index = 0
            self.__fp = open(ds.path, 'r')

        def __next__(self):
            if(self.__index < self.__ds.length):
                line = self.__fp.readline()
                line = json.loads(line.rstrip())
                self.__index += 1
                if(self.__index == self.__ds.length):
                    self.__fp.close()
                return line
            raise StopIteration
