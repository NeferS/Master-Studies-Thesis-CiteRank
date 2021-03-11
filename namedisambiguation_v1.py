"""Name Disambiguation

Requires
--------
dblputils, textdistance, numpy, scipy, sklearn

Classes
-------
AuthorNameDisambiguation :
    A class that can be used to solve the task of author names disambiguation.

MergeInfo :
    A simple class that can be used to store a new authors' dataset which merges
    the duplicates computed by the class AuthorNameDisambiguation and to compute
    a dictionary where each author's id is mapped to itself or to its duplicate.
"""

import os
import math
from collections.abc import Iterable
from difflib import SequenceMatcher
from timeit import default_timer as timer

from dblputils import IOUtil
from textdistance import RatcliffObershelp
import numpy as np
from scipy import sparse
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer

class AuthorNameDisambiguation:
    """
    This class is used to solve the task of author names disambiguation. This
    task is solved in three steps: recall step (r_step), precision step (p_step)
    and merge step (m_step). In the first step a list of possible duplicates is
    computed for each author based on their name string similarity; in the second
    step some similarity matrices are computed to determine the similarity
    between two authors based on meta-paths; in the third (and last step) the
    authors pairs with a similarity score greater than a threshold (meta-path
    based) are considered to be the same person and are merged together.

    Attributes
    ----------
    duplicates : dict
        a dictionary where for each <key,value> pair the key is an author's id
        and the value is the list of his duplicates resulting from the m_step
    ioutil : IOUtil
        an istance of the IOUtil utility class used to do some I/O operations
    Map : sparse.csr_matrix
        the matrix Map (author-publication) where a_ij=1 if the author i wrote
        the publication j, a_ij=0 otherwise
    Mapapa : sparse.csr_matrix
        the matrix associated to meta-path APAPA
    Mat : sparse.csr_matrix
        the dot product between the matrices Map and Mpt, where Mpt is the matrix
        (publication-title) where a_ij=n, the word j appears n times in the title
        of publication i
    Mav : sparse.csr_matrix
        the matrix Mav (author-venue) where a_ij=n, the author i published n
        times on venue j
    May : sparse.csr_matrix
        the dot product between the matrices Map and Mpy, where Mpy is the matrix
        (publication-year) where a_ij=1 if the publication i was released on year
        j, a_ij=0 otherwise
    sims : dict
        a dictionary where for each <key,value> pair the key is the author's id
        of an author that has at least one possible duplicate (from r_step) and
        the value is a list where the i-th element is the weighted sum of
        similarity values (one for each meta-path) between the author and his
        i-th possible duplicate
    possible_duplicates : dict
        a dictionary where for each <key,value> pair the key is an author's id
        and the value is the list of his possible duplicates resulting from the
        r_step

    Methods
    -------
    computeMap(auth_id_ind:dict, pub_id_ind:dict, publications_dict:dict, filename:str=None) -> None
        Computes the Map matrix used to compute most of the other matrices.

    computeMapapa(filename:str=None) -> None
        Computes the Mapapa matrix used in the p_step.

    computeMat(titles_list:Iterable, filename:str=None) -> None
        Computes the Mat matrix used in the p_step.

    computeMav(authors:Iterable, venues_dict:dict, auth_id_ind:dict, filename:str=None) -> None
        Computes the Mav matrix used in the p_step.

    computeMay(publications:Iterable, pub_id_ind:dict, years_dict:dict, filename:str=None) -> None
        Computes the May matrix used in the p_step.

    loadDuplicates(filename:str) -> None
        Loads the duplicates resulting from the merge step from file.

    loadMatrices(**kwargs) -> None
        Loads the values of the matrices from files.

    loadSims(filename:str) -> None
        Loads the PathSim values resulting from the precision step from file.

    loadPossibleDuplicates(filename:str) -> None
        Loads the possible duplicates resulting from the recall step from file.

    m_step(threshold:float=0.55, filename:str=None) -> None
        Computes the merge step: finding the actual duplicates for each
        authors pair that passed the first steps, i.e. the possible duplicate
        authors pairs with a similarity value (meta-path based) over a threshold
        are considered to be actual duplicates.

    p_step(w:Iterable, auth_id_ind:dict, filename:str=None) -> None
        Computes the precision step: the PathSim values are computed based on
        meta-paths AVA (same venue), APAPA (co-authors of my co-authors), APTPA
        (title similarities), APYPA (same publication year).

    r_step(authors_dict:dict, filename:str, start:int=0, thresholds:Iterable=[0.5,0.85,0.85]) -> None
        Computes the recall step: for each author find all possible duplicates
        using RatcliffObershelp similarity as criteria.
    """
    def __init__(self, ioutil:IOUtil):
        if(not isinstance(ioutil, IOUtil)):
            raise TypeError("ioutil must be an istance of IOUtil but %s was passed"%str(type(ioutil)))

        self.ioutil = ioutil
        # Possible duplicates dict structure:
        # dict := {<author_id>:<matches>}
        #    <author_id> := str
        #    <matches> := list<str>
        self.possible_duplicates = None
        self.Map = None
        self.Mapapa = None
        self.May = None
        self.Mat = None
        self.Mav = None
        self.sims = None
        self.duplicates = None

        self.__ratcliff = RatcliffObershelp()
        self.__l2_norm = Normalizer(norm='l2').transform
        self.__attr_names = ["Map","Mapapa","May","Mat","Mav"]

    def __quick_compare(self, s0:str, s1:str) -> float :
        # A quick ratio used to determine if two authors' name strings could match;
        # in this case the initials of the names are compared
        tks0 = s0.split()
        tks1 = s1.split()
        matches = sum([tk0[0]==tk1[0] for tk1 in tks1 for tk0 in tks0])
        return matches/(len(tks0)+len(tks1))

    def r_step(self, authors_dict:dict, filename:str, start:int=0, thresholds:Iterable=[0.5,0.85,0.85]) -> None :
        """
        Computes the recall step (step 1): for each author find all possible
        duplicates using RatcliffObershelp similarity as criteria. Since the
        computation may take a lot of time (O(n^2)), each author match is saved
        when computed, so the computation can be stopped and resumed, if needed.

        Parameters
        ----------
        authors_dict : dict
            a dictionary where each pair should have an author's id as key and
            his name as value
        filename : str
            the file where the possible duplicates have to be stored
        start : int, optional
            the index of the first author that has to be considered (default is 0
            if the computation must start from the beginning)
        thresholds : Iterable, optional
            the thresholds that must be reached by the similarity measures; the
            values, from first to last, refer to initals similarity,
            RatcliffObershelp similarity upper bound and RatcliffObershelp
            similarity (default is [0.5,0.85,0.85])

        Returns
        -------
        None
            the possible duplicates are stored in the class attribute
            "self.possible_duplicates"
        """
        if(not isinstance(authors_dict,dict)):
            raise TypeError("authors_dict must be a dictionary but %s was passed"%str(type(authors_dict)))
        if(not isinstance(start,int)):
            raise TypeError("start must be an integer but %s was passed"%str(type(start)))
        if(not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))
        if(start<0 or start>len(authors_dict)):
            raise ValueError("start must be an integer between 0 and len(authors_dict) but %d was passed"%start)
        if(not isinstance(thresholds,Iterable)):
            raise TypeError("thresholds must be an Iterable but %s was passed"%str(type(thresholds)))

        if(start==0):
            self.possible_duplicates = dict()
        else:
            self.loadPossibleDuplicates(filename)
        # A list of seen names is also stored: if a name is completely equal to
        # an already seen one (homonim), than the iteration is avoided and the
        # list of possible duplicates is copied from the homonim.
        seen_names = dict()
        if(len(self.possible_duplicates)>0):
            for idx in self.possible_duplicates:
                seen_names[authors_dict[idx]] = idx

        ordinal_auth_ids = list(authors_dict.keys())
        opening_type = 'w' if start==0 else 'a'
        start_time = timer()
        with open(filename, opening_type) as file:
            # for each author
            for i in range(start, len(authors_dict)):
                id0 = ordinal_auth_ids[i]
                auth_name = authors_dict[id0]
                # an homonim has been found
                if(auth_name in seen_names):
                    duplicates = self.possible_duplicates[seen_names[auth_name]].copy() #copy the possible duplicates from the homonim
                    if(id0 in duplicates):
                        index = duplicates.index(id0) #remove this author from the list
                        duplicates[index] = seen_names[auth_name] #add the homonim to this author's list of possible duplicates
                    else:
                        duplicates.append(seen_names[auth_name]) #add the homonim to this author's list of possible duplicates
                    self.possible_duplicates[id0] = duplicates
                    file.write("%s;%s%s"%(id0,self.possible_duplicates[id0],"" if i==len(authors_dict)-1 else "\n"))
                    continue
                duplicates = []
                # for each other author after it
                for j in range(i+1, len(authors_dict)):
                    id1 = ordinal_auth_ids[j]
                    # first the initials of the authors are tested (at least two common initials)
                    if(self.__quick_compare(auth_name, authors_dict[id1])>=thresholds[0]):
                        # if the first test passes, the two author's names similarity is tested by a quick ratio
                        # (RatcliffObershelp upper bound)
                        if(SequenceMatcher(a=auth_name, b=authors_dict[id1]).quick_ratio()>thresholds[1]):
                            # if the strings match, their similarity is computed by RatcliffObershelp similarity
                            if(self.__ratcliff.similarity(auth_name, authors_dict[id1])>thresholds[2]):
                                duplicates.append(id1)
                self.possible_duplicates[id0] = duplicates
                seen_names[auth_name] = id0
                file.write("%s;%s%s"%(id0,self.possible_duplicates[id0],"" if i==len(authors_dict)-1 else "\n"))
                if(i%10==0):
                    end = timer()
                    elapsed = round(end-start_time,2)
                    mins = math.floor(elapsed/60)
                    secs = math.ceil(elapsed-mins*60)
                    print("\rExamined: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i,len(authors_dict),mins,secs),end='',flush=True)
            end = timer()
            elapsed = round(end-start_time,2)
            mins = math.floor(elapsed/60)
            secs = math.ceil(elapsed-mins*60)
            print("\rExamined: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i+1,len(authors_dict),mins,secs))

    def computeMap(self, auth_id_ind:dict, pub_id_ind:dict, publications_dict:dict, filename:str=None) -> None :
        """
        Computes the Map (author-publication) matrix, the matrix where a_ij=1 if
        the author i wrote the publication j, a_ij=0 otherwise. This matrix is
        used to compute most of the other matrices.

        Parameters
        ----------
        auth_id_ind : dict
            a dictionary where each pair should have an author's id as key and
            his ordinal integer as value (used as index in the matrix)
        pub_id_ind : dict
            a dictionary where each pair should have the id of a publication as
            key and its ordinal integer as value (used as index in the matrix)
        publications_dict : dict
            a dictionary where each pair should have the id of a publication as
            key and the list of its authors as value
        filename : str, optional
            if it is not None, the matrix will be saved in the file with filename
            (default is None)

        Returns
        -------
        None
            the matrix is stored in the class attribute "self.Map"
        """
        if(not isinstance(auth_id_ind,dict)):
            raise TypeError("auth_id_ind must be a dict but %s was passed"%str(type(auth_id_ind)))
        if(not isinstance(pub_id_ind,dict)):
            raise TypeError("pub_id_ind must be a dict but %s was passed"%str(type(pub_id_ind)))
        if(not isinstance(publications_dict,dict)):
            raise TypeError("publications_dict must be a dict but %s was passed"%str(type(publications_dict)))
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        self.Map = sparse.lil_matrix((len(auth_id_ind),len(publications_dict)), dtype=np.int8) #to csr later
        start = timer()
        for pub in publications_dict:
            auths = publications_dict[pub]
            j = pub_id_ind[pub]
            for auth in auths:
                i = auth_id_ind[auth]
                self.Map[i,j] = 1
            if(j%10000==0):
                print("\rCompleted: %d/%d"%(j,len(publications_dict)),end='',flush=True)
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rCompleted: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(j+1,len(publications_dict),mins,secs))
        self.Map = self.Map.tocsr()
        if(filename is not None):
            sparse.save_npz(filename, self.Map)

    def computeMapapa(self, filename:str=None) -> None:
        """
        Computes the Mapapa matrix used in the p_step. This matrix stores the
        similarity values between authors based on meta-path APAPA.

        Parameters
        ----------
        filename : str, optional
            if it is not None, the matrix will be saved in the file with filename
            (default is None)

        Returns
        -------
        None
            the matrix is stored in the class attribute "self.Mapapa"
        """
        # Step 2.1: computing meta-path APAPA for each author (applying l2-norm)
        if(self.Map is None):
            raise NotImplementedError("can't compute matrix Mapapa if Map is None")
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        start = timer()
        Maa = self.__l2_norm(self.Map.dot(self.Map.T))
        self.Mapapa = Maa.dot(Maa.T)
        if(filename is not None):
            sparse.save_npz(filename, self.Mapapa)
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("Elapsed: %d min(s) %d sec(s)"%(mins,secs))

    def computeMay(self, publications:Iterable, pub_id_ind:dict, years_dict:dict, filename:str=None) -> None :
        """
        Computes the May matrix used in the p_step. The dot product between May
        and May.T stores the similarity values between authors based on meta-path
        APYPA.

        Parameters
        ----------
        publications : Iterable
            a list of json objects representing a publication
        pub_id_ind : dict
            a dictionary where each pair should have the id of a publication as
            key and its ordinal integer as value (used as index in the matrix)
        years_dict : dict
            a dictionary where each pair should have a year of publication as
            key and an ordinal integer associated to the year as value (used as
            index in the matrix)
        filename : str, optional
            if it is not None, the matrix will be saved in the file with filename
            (default is None)

        Returns
        -------
        None
            the matrix is stored in the class attribute "self.May"
        """
        #2.2.1: computing the adjacency matrix Mpy (publication-year) where a_ij=1 if the publication i was
        # released on year j, a_ij=0 otherwise
        if(self.Map is None):
            raise NotImplementedError("can't compute matrix May if Map is None")
        if(not isinstance(publications,Iterable)):
            raise TypeError("publications must be an Iterable but %s was passed"%str(type(publications)))
        if(not isinstance(pub_id_ind,dict)):
            raise TypeError("pub_id_ind must be a dict but %s was passed"%str(type(pub_id_ind)))
        if(not isinstance(years_dict,dict)):
            raise TypeError("years_dict must be a dict but %s was passed"%str(type(years_dict)))
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        Mpy = sparse.lil_matrix((len(publications),len(years_dict)), dtype=np.int8) # to csr later
        start = timer()
        for jsonPub in publications:
            i = pub_id_ind[jsonPub['id']]
            year = jsonPub['year']
            if(year!=''):
                j = years_dict[str(year)]
                Mpy[i,j] = 1
            if(i%10000==0):
                print("\rRows completed: %d/%d"%(i,len(publications)),end='',flush=True)
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rRows completed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i+1,len(publications),mins,secs))
        #2.2.2: computing Mapypa
        self.May = self.__l2_norm(self.Map.dot(Mpy.tocsr()))
        # original operation is too expensive in terms of memory, better to do inline products
        #May = convert_to_64bit_indices(May)
        #Mapypa = May.dot(May.T)
        if(filename is not None):
            sparse.save_npz(filename, self.May)

    def computeMat(self, titles_list:Iterable, filename:str=None) -> None:
        """
        Computes the Mat matrix used in the p_step. The dot product between Mat
        and Mat.T stores the similarity values between authors based on meta-path
        APTPA.

        Parameters
        ----------
        titles_list : Iterable
            the list of titles of the publications
        filename : str, optional
            if it is not None, the matrix will be saved in the file with filename
            (default is None)

        Returns
        -------
        None
            the matrix is stored in the class attribute "self.Mat"
        """
        if(self.Map is None):
            raise NotImplementedError("can't compute matrix Mat if Map is None")
        if(not isinstance(titles_list,Iterable)):
            raise TypeError("titles_list must be an Iterable but %s was passed"%str(type(titles_list)))
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        #2.3.1: create the matrix Mpt (publication-title) where a_ij=n, the word j appears n times in the title of publication i
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(titles_list)
        #2.3.2: computing Maptpa
        self.Mat = self.__l2_norm(self.Map.dot(X))
        # original operation is too expensive in terms of memory, better to do inline products
        #Mat = convert_to_64bit_indices(Mat)
        #Maptpa = Mat.dot(Mat.T)
        if(filename is not None):
            sparse.save_npz(filename, self.Mat)

    def computeMav(self, authors:Iterable, venues_dict:dict, auth_id_ind:dict, filename:str=None) -> None :
        """
        Computes the Mav matrix used in the p_step. The dot product between Mav
        and Mav.T, after some other operations, stores the similarity values
        between authors based on meta-path AVA.

        Parameters
        ----------
        authors : Iterable
            a list of json objects representing an author
        years_dict : dict
            a dictionary where each pair should have a publication venue as key
            and an ordinal integer associated to the venue as value (used as
            index in the matrix)
        auth_id_ind : dict
            a dictionary where each pair should have an author's id as key and
            his ordinal integer as value (used as index in the matrix)
        filename : str, optional
            if it is not None, the matrix will be saved in the file with filename
            (default is None)

        Returns
        -------
        None
            the matrix is stored in the class attribute "self.Mav"
        """
        if(not isinstance(authors,Iterable)):
            raise TypeError("authors must be an Iterable but %s was passed"%str(type(authors)))
        if(not isinstance(auth_id_ind,dict)):
            raise TypeError("auth_id_ind must be a dict but %s was passed"%str(type(auth_id_ind)))
        if(not isinstance(venues_dict,dict)):
            raise TypeError("venues_dict must be a dict but %s was passed"%str(type(venues_dict)))
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        #2.4.1: computing the adjacency matrix Mav (author-venue) where a_ij=n, the author i published n times on venue j
        self.Mav = sparse.lil_matrix((len(authors), len(venues_dict)), dtype=np.float) # to csr later
        start = timer()
        for author in authors:
            i = auth_id_ind[author['id']]
            pubs_list = author['pubs']
            for pub in pubs_list:
                venue = pub['venue']
                if(venue!=""):
                    j = venues_dict[venue]
                    self.Mav[i,j] += 1
            if(i%10000==0):
                print("\rRows completed: %d/%d"%(i,len(authors)),end='',flush=True)
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rRows completed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i+1,len(authors),mins,secs))
        self.Mav = self.Mav.tocsr()
        #2.4.2: computing Mava
        # too expensive in terms of memory and time, better to do inline computations
        #Mav = convert_to_64bit_indices(Mav.tocsr())
        #M = Mav.dot(Mav.T)
        #diagonal = sparse.csc_matrix(M.diagonal(), dtype=np.float).T #diagonal+diagonal.T is the denominator of PathSim for Mava elements
        #M = 2*M # numerator of PathSim for Mava elements
        #diagonal = diagonal.tolil()
        #M = M.tolil()
        #for i in range(authors_lines):
        #    for j in range(authors_lines):
        #        M[i,j] /= diagonal[i,0]+diagonal[j,0]
        if(filename is not None):
            sparse.save_npz(filename, self.Mav)

    # Fixes scipy RuntimeError during dot computation: 'nnz of the result is too large' (UNUSED)
    #def convert_to_64bit_indices(A):
    #    A.indptr = np.array(A.indptr, copy=False, dtype=np.int64)
    #    A.indices = np.array(A.indices, copy=False, dtype=np.int64)
    #    return A

    def p_step(self, w:Iterable, auth_id_ind:dict, filename:str=None) -> None :
        """
        Computes the precision step (step 2): the Sim values are computed
        based on meta-paths AVA (same venue), APAPA (co-authors of my co-authors),
        APTPA (title similarities), APYPA (same publication year). The sum of
        similarity values of each meta-path is the resulting similarity score
        for each author and one of his possible duplicates.

        Parameters
        ----------
        w : Iterable
            a vector with a weight for each similarity value based on a meta-path;
            the position are used as follows: { 0:APAPA, 1:APYPA, 2: APTPA, 3:AVA }
        auth_id_ind : dict
            a dictionary where each pair should have an author's id as key and
            his ordinal integer as value (used as index in the matrices)
        filename : str, optional
            if it is not None, the dictionary will be saved in the file with
            filename (default is None)

        Returns
        -------
        None
            the dictionary is stored in the class attribute "self.sims"
        """
        if(self.possible_duplicates is None):
            raise NotImplementedError("can't compute p_step if possible_duplicates is None")
        if(self.Mapapa is None):
            raise NotImplementedError("can't compute p_step if Mapapa is None")
        if(self.May is None):
            raise NotImplementedError("can't compute p_step if May is None")
        if(self.Mat is None):
            raise NotImplementedError("can't compute p_step if Mat is None")
        if(self.Mav is None):
            raise NotImplementedError("can't compute p_step if Mav is None")
        if(not isinstance(w,Iterable)):
            raise TypeError("w must be an Iterable but %s was passed"%str(type(w)))
        if(not isinstance(auth_id_ind,dict)):
            raise TypeError("auth_id_ind must be a dict but %s was passed"%str(type(auth_id_ind)))
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        pd_hasValues = self.possible_duplicates.copy()
        for key in self.possible_duplicates:
            if(self.possible_duplicates[key]==[]):
                del pd_hasValues[key]
        self.sims = dict()
        i = 0
        start = timer()
        for id0 in pd_hasValues:
            ind0 = auth_id_ind[id0]
            sims = [0]*len(pd_hasValues[id0])
            for j in range(len(pd_hasValues[id0])):
                ind1 = auth_id_ind[pd_hasValues[id0][j]]
                values = np.zeros(4)
                values[0] = self.Mapapa[ind0, ind1]
                values[1] = self.May.getrow(ind0).dot(self.May.T.getcol(ind1)).toarray()[0][0]
                values[2] = self.Mat.getrow(ind0).dot(self.Mat.T.getcol(ind1)).toarray()[0][0]
                denom = self.Mav.getrow(ind0).dot(self.Mav.T.getcol(ind0)).toarray()[0][0] + self.Mav.getrow(ind1).dot(self.Mav.T.getcol(ind1)).toarray()[0][0]
                num = 2*self.Mav.getrow(ind0).dot(self.Mav.T.getcol(ind1)).toarray()[0][0]
                values[3] = num/denom
                sims[j] = values.dot(w)
            self.sims[id0] = sims
            if(i%100==0):
                end = timer()
                elapsed = round(end-start,2)
                mins = math.floor(elapsed/60)
                secs = math.ceil(elapsed-mins*60)
                print("\rComputed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i,len(pd_hasValues),mins,secs),end='',flush=True)
            i += 1
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rComputed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i,len(pd_hasValues),mins,secs))
        if(filename is not None):
            with open(filename, 'w') as file:
                i = 0
                for key in self.sims:
                    file.write("%s;%s%s"%(key,json.dumps(self.sims[key]),("" if i==len(self.sims)-1 else "\n")))
                    i += 1

    def m_step(self, threshold:float=0.55, filename:str=None) -> None :
        """
        Computes the merge step (step 3): finding the actual duplicates for each
        authors pair that passed the first step, i.e. the possible duplicate
        authors pairs with a similarity value (meta-path based) over a threshold
        are considered to be actual duplicates.

        Parameters
        ----------
        threshold : float, optional
            used to determine wheter two authors are duplicates or not, i.e. if
            their PathSim value is over this value (default is 0.55)
        filename : str, optional
            if it is not None, the dictionary will be saved in the file with
            filename (default is None)

        Returns
        -------
        None
            the dictionary is stored in the class attribute "self.duplicates"
        """
        if(self.possible_duplicates is None):
            raise NotImplementedError("can't compute the merge step if possible_duplicates is None")
        if(self.sims is None):
            raise NotImplementedError("can't compute the merge step if sims is None")
        if(not isinstance(threshold,float)):
            raise TypeError("threshold must be a float but %s was passed"%str(type(threshold)))
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))
        if(threshold<=0.0 or threshold>1.0):
            raise ValueError("threshold must be a float greater than 0.0 and lower or equal to 1.0, but %s was passed"%str(threshold))

        pd_hasValues = self.possible_duplicates.copy()
        for key in self.possible_duplicates:
            if(self.possible_duplicates[key]==[]):
                del pd_hasValues[key]
        # Step 3.0: finding the actual duplicates for each author that passed
        # the first step, i.e. the possible duplicate authors with a similarity
        # value over a threshold
        self.duplicates = dict()
        j = 0
        start = timer()
        # for each author key from the list of keys with at least one possible
        # duplicate
        for key in self.sims:
            # for each value in the list of its possible duplicates
            for i in range(len(self.sims[key])):
                # it is a confirmed duplicate
                if(self.sims[key][i]>=threshold):
                    # takes the key of the duplicate author
                    duplicate_key = pd_hasValues[key][i]
                    if(not key in self.duplicates):
                        self.duplicates[key] = [duplicate_key]
                    else:
                        self.duplicates[key].append(duplicate_key)
            if(j%100==0):
                print("\rComputed: %d/%d"%(j,len(self.sims)),end='',flush=True)
            j += 1
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rComputed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(j,len(self.sims),mins,secs))
        # Step 3.1: merging matching author ids that have different entries in
        # the duplicates dictionary, e.g. if the entry corresponding to author
        # id "000" has the id "001" as a duplicate and the id "001" is an entry
        # with duplicate "010", then the id "010" is added to the list of the id
        # "000" and the entry relative to "001" is removed. This is done
        # iteratively in order to be sure that each of the entries of this kind
        # will be removed.
        current = self.duplicates
        next_dict = self.duplicates.copy()
        removed = []
        iterations = 0
        flag = True
        while(flag):
            flag = False
            for key0 in current:
                if(key0 in removed):
                    continue
                for key1 in current:
                    if(key0==key1 or key1 in removed):
                        continue
                    if(key1 in current[key0]):
                        set_key1 = set(current[key1])
                        if(key0 in set_key1):
                            set_key1.remove(key0)
                        next_dict[key0] = list(set(next_dict[key0])|set_key1)
                        del next_dict[key1]
                        removed.append(key1)
                        flag = True
            current = next_dict
            next_dict = next_dict.copy()
            iterations += 1
            print("\rIterations: %d"%iterations,end='',flush=True)
        print("\rIterations: %d"%iterations)
        self.duplicates = next_dict
        if(filename is not None):
            with open(filename, 'w') as file:
                i = 0
                for key in self.duplicates:
                    file.write("%s;%s%s"%(key,self.duplicates[key],("" if i==len(self.duplicates)-1 else "\n")))
                    i += 1

    def loadPossibleDuplicates(self, filename:str) -> None :
        """
        Loads the possible duplicates resulting from the recall step from file.

        Parameters
        ----------
        filename : str
            the file where the possible duplicates are stored

        Returns
        -------
        None
            the result is stored in the class attribute "self.possible_duplicates"
        """
        if(not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))
        if(not os.path.exists(filename) or not os.path.isfile(filename)):
            raise ValueError("invalid filename")

        self.possible_duplicates = self.ioutil.loadKeysLists(filename, ";")

    def loadSims(self, filename:str) -> None :
        """
        Loads the Sim values resulting from the precision step from file.

        Parameters
        ----------
        filename : str
            the file where the Sims values are stored

        Returns
        -------
        None
            the result is stored in the class attribute "self.sims"
        """
        if(not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))
        if(not os.path.exists(filename) or not os.path.isfile(filename)):
            raise ValueError("invalid filename")

        self.sims = dict()
        with open(filename, 'r') as file:
            line = file.readline()
            while(line!=''):
                line = line.rstrip()
                splt = line.split(";")
                self.sims[splt[0]] = json.loads(splt[1])
                line = file.readline()

    def loadDuplicates(self, filename:str) -> None :
        """
        Loads the duplicates resulting from the merge step from file.

        Parameters
        ----------
        filename : str
            the file where the duplicates are stored

        Returns
        -------
        None
            the result is stored in the class attribute "self.duplicates"
        """
        if(not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))
        if(not os.path.exists(filename) or not os.path.isfile(filename)):
            raise ValueError("invalid filename")

        self.duplicates = self.ioutil.loadKeysLists(filename, ";")

    def loadMatrices(self, **kwargs) -> None :
        """
        Loads the values of the specified matrices from files. In order to load
        a specific matrix from a file, it's required to pass a kwarg with value
        set to the path of the file which contains the matrix (e.g. Map="./Map.npz"
        to load the matrix Map).

        Parameters
        ----------
        **kwargs :
            one (or more) from { Map,Mapapa,May,Mat,Mav }; other kwargs will be
            simply ignored without raising any error

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
                if(attr=="Map"):
                    self.Map = sparse.load_npz(path)
                elif(attr=="Mapapa"):
                    self.Mapapa = sparse.load_npz(path)
                elif(attr=="May"):
                    self.May = sparse.load_npz(path)
                elif(attr=="Mat"):
                    self.Mat = sparse.load_npz(path)
                else: #attr=="Mav"
                    self.Mav = sparse.load_npz(path)
                print("%s attribute loaded"%attr)

class MergeInfo:
    """
    This simple class can be used to store a new authors' dataset which merges
    the duplicates computed by the class AuthorNameDisambiguation and to compute
    a dictionary where each author's id is mapped to itself or to its duplicate.

    Attributes
    ----------
    idMap : dict
        dictionary where for each <key,value> pair the key is an author's id
        from the complete authors dataset and the value is the same id if he's
        no one duplicate or his duplicate's id if he's someone's other duplicate
    ioutil : IOUtil
        an istance of the IOUtil utility class used to do some I/O operations

    Methods
    -------
    computeIdMap(authors:Iterable, duplicates:dict, filename:str=None) -> None
        Computes the idMap dictionary.

    computeMergeDataset(authors:dict, pubs_dict:dict, duplicates:dict, filename:str=None) -> Iterable
        Merges the duplicate authors in the original dataset and stores the new
        dataset in a new Iterable where each entry is a json object representing
        an author.

    loadIdMap(filename:str) -> None
        Loads the idMap from file.
    """
    def __init__(self, ioutil:IOUtil):
        if(not isinstance(ioutil, IOUtil)):
            raise TypeError("ioutil must be an istance of IOUtil but %s was passed"%str(type(ioutil)))

        self.ioutil = ioutil
        self.idMap = None

    def computeIdMap(self, authors:Iterable, duplicates:dict, filename:str=None) -> None :
        """
        Computes the idMap dictionary. In the resulting dictionary each pair have
        an author's id from the complete authors dataset as key and the same id
        if he's no one duplicate or his duplicate's id if he's someone's other
        duplicate as value.

        Parameters
        ----------
        authors : Iterable
            a list of json objects representing an author
        duplicates : dict
            a dictionary where each pair should have an author's id as key and
            his list of duplicates' ids as value
        filename : str, optional
            if it is not None, the dictionary will be saved in the file with
            filename (default is None)

        Returns
        -------
        None
            the dictionary is stored in the class attribute "self.idMap"
        """
        if(not isinstance(authors,Iterable)):
            raise TypeError("authors must be an Iterable but %s was passed"%str(type(authors)))
        if(not isinstance(duplicates,dict)):
            raise TypeError("duplicates must be a dict but %s was passed"%str(type(duplicates)))
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        self.idMap = dict()
        i = 0
        for author in authors:
            self.idMap[author['id']] = author['id']
            if(i%10000==0):
                print("\rCompleted: %d/%d"%(i,len(authors)),end='',flush=True)
            i += 1
        print("\rCompleted: %d/%d"%(i,len(authors)))
        i = 0
        for idx in duplicates:
            for dupl in duplicates[idx]:
                self.idMap[dupl] = idx
            if(i%100==0):
                print("\rMapped: %d/%d"%(i,len(duplicates)),end='',flush=True)
            i += 1
        print("\rMapped: %d/%d"%(i,len(duplicates)))
        if(filename is not None):
            with open(filename, 'w') as file:
                i = 0
                for idx in self.idMap:
                    file.write("%s;%s%s"%(idx,self.idMap[idx],("" if i==len(self.idMap)-1 else "\n")))
                    if(i%10000==0):
                        print("\rLines written: %d/%d"%(i,len(self.idMap)),end='',flush=True)
                    i += 1
                print("\rLines written: %d/%d"%(i,len(self.idMap)))

    def computeMergeDataset(self, authors:dict, pubs_dict:dict, duplicates:dict, filename:str=None) -> Iterable :
        """
        Merges the duplicate authors in the original dataset and stores the new
        dataset in a new Iterable where each entry is a json object representing
        an author (e.g. if the author with id "000" has a duplicate with id "001",
        the second's publications are added to the first's, then the second is
        deleted from the dataset).

        Parameters
        ----------
        authors : dict
            a dictionary where each pair shoud have an author's id as key and a
            json object representing the author as value
        pubs_dict : dict
            a dictionary where each pair should have an author's id as key and
            the list of his publication ids as value
        duplicates : dict
            a dictionary where each pair should have an author's id as key and
            his list of duplicates' ids as value
        filename : str, optional
            if it is not None, the dataset will be saved in the file with filename
            (default is None)

        Returns
        -------
        Iterable
            the resulting dataset as an Iterable, where each entry is a json
            object representing an author
        """
        if(not isinstance(authors,dict)):
            raise TypeError("authors must be a dict but %s was passed"%str(type(authors)))
        if(not isinstance(duplicates,dict)):
            raise TypeError("duplicates must be a dict but %s was passed"%str(type(duplicates)))
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        seen = []
        dataset = []
        for key in authors:
            # already seen and merged
            if(key in seen):
                continue
            # has at least one duplicate
            if(key in duplicates):
                jsonObj = authors[key]
                # for each of its duplicates
                for duplicate in duplicates[key]:
                    pubs = authors[duplicate]['pubs']
                    # for each publication of the duplicate author
                    for pub in pubs:
                        if(pub['id'] not in pubs_dict[key]):
                            jsonObj['pubs'].append(pub)
                    seen.append(duplicate)
                dataset.append(jsonObj)
                continue
            # not seen and has not any duplicate
            dataset.append(authors[key])
        if(filename is not None):
            self.ioutil.dumpLinesFromJson(dataset, filename)
        return dataset

    def loadIdMap(self, filename:str) -> None :
        """
        Loads the idMap from file.

        Parameters
        ----------
        filename : str
            the file where the idMap is stored

        Returns
        -------
        None
            the result is stored in the class attribute "self.idMap"
        """
        if(not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))
        if(not os.path.exists(filename) or not os.path.isfile(filename)):
            raise ValueError("invalid filename")

        self.idMap = self.ioutil.loadKeysVals(filename, ";")
