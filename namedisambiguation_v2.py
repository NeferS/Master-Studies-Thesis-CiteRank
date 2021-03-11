"""Name Disambiguation

Requires
--------
dblputils, json, textdistance, numpy, scipy, sklearn

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
import multiprocessing
from threading import Thread, Semaphore
from timeit import default_timer as timer

from dblputils import IOUtil
import json
from textdistance import RatcliffObershelp
import numpy as np
from scipy import sparse
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer

class AuthorNameDisambiguation:
    """
    This class is used to solve the task of author names disambiguation. This
    task is solved in three steps: recall step (r_step), precision step (p_step),
    and merge step (m_step). In the first step a similarity matrix based on
    meta-path similarities is computed; in the second step the similarity based
    on string matching is computed for the non zero elements of the previously
    computed matrix; in the third (and last step) the authors pairs with a
    similarity score greater than a threshold (string-based) are considered to be
    the same person and are merged together.

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
    Mav_diagonal : numpy.array
        an array containing the diagonal elements of the matrix which is the
        result of the dot product between the matrices Mav and Mav.T
    May : sparse.csr_matrix
        the dot product between the matrices Map and Mpy, where Mpy is the matrix
        (publication-year) where a_ij=1 if the publication i was released on year
        j, a_ij=0 otherwise
    sims : sparse.csr_matrix
        a sparse matrix where a_ij = sim(i,j) if sim(i,j)>threshold, 0.0 otherwise;
        sim(i,j) is the similarity value between the pair of authors (i,j)
    string_sims : dict
        a dictionary where for each <key,value> pair the key is an author's id of
        an author that has at least one possible duplicate (from p_step) and the
        value is a list where the i-th element is a pair <id,sim(i,j)>, where id
        is the id of the i-th possible duplicate of the author and sim(i,j) is
        the value of the string-based similarity between the pair of authors

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

    computeMavDiagonal(self, filename:str=None) -> None
        Computes the diagonal of the matrix result from the dot product between
        matrices Mav and Mav.T

    computeMay(publications:Iterable, pub_id_ind:dict, years_dict:dict, filename:str=None) -> None
        Computes the May matrix used in the p_step.

    loadDuplicates(filename:str) -> None
        Loads the duplicates resulting from the merge step from file.

    loadMatrices(**kwargs) -> None
        Loads the values of the matrices from files.

    loadStringSims(filename:str) -> None
        Loads the similarity values based on string matching from file.

    m_step(threshold:float=0.55, filename:str=None) -> None
        Computes the merge step: finding the actual duplicates for each
        authors pair that passed the first steps, i.e. the possible duplicate
        authors pairs with a similarity value (string based) over a threshold are
        considered to be actual duplicates.

    p_step(w:Iterable, n_workers:int=4, auth_id_ind:dict, filename:str=None) -> None
        Computes the precision step: for each author that has at least one
        possible duplicate from recall step, compute the string similarity
        based on RatcliffObershelp criteria.

    r_step(authors_dict:dict, filename:str, start:int=0) -> None
        Computes the recall step: the Sim values are computed based on meta-paths
        AVA (same venue), APAPA (co-authors of my co-authors), APTPA (title
        similarities), APYPA (same publication year).
    """
    def __init__(self, ioutil:IOUtil):
        if(not isinstance(ioutil, IOUtil)):
            raise TypeError("ioutil must be an istance of IOUtil but %s was passed"%str(type(ioutil)))

        self.ioutil = ioutil
        self.Map = None
        self.Mapapa = None
        self.May = None
        self.Mat = None
        self.Mav = None
        self.Mav_diagonal = None
        self.sims = None
        self.string_sims = None
        self.duplicates = None

        self.__ratcliff = RatcliffObershelp()
        self.__l2_norm = Normalizer(norm='l2').transform
        self.__attr_names = ["Map","Mapapa","May","Mat","Mav","Mav_diagonal","Sims"]

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
        vectorizer = TfidfVectorizer()
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

    def computeMavDiagonal(self, filename:str=None) -> None :
        """
        Computes the diagonal of the matrix result from the dot product between
        matrices Mav and Mav.T.

        Parameters
        ----------
            filename : str, optional
                if it is not None, the diagonal will be saved in the file with
                filename (default is None)

        Returns
        -------
        None
            the result is stored in the class attribute "self.Mav_diagonal"
        """
        if(self.Mav is None):
            raise NotImplementedError("can't compute Mav_diagonal if Mav is None")
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        self.Mav_diagonal = np.zeros(self.Mav.shape[0])
        for i in range(self.Mav.shape[0]):
            self.Mav_diagonal[i] = self.Mav.getrow(i).dot(self.Mav.T.getcol(i)).toarray()[0][0]
            if(i%1000==0):
                print("\r%d/%d"%(i,self.Mav.shape[0]),end='',flush=True)
        print("\r%d/%d"%(i+1,self.Mav.shape[0]))
        if(filename is not None):
            np.save(filename, self.Mav_diagonal)

    # Fixes scipy RuntimeError during dot computation: 'nnz of the result is too large' (UNUSED)
    #def convert_to_64bit_indices(A):
    #    A.indptr = np.array(A.indptr, copy=False, dtype=np.int64)
    #    A.indices = np.array(A.indices, copy=False, dtype=np.int64)
    #    return A

    def __computeSims(self, w, first, last, zeros, semaphore, threshold):
        # Concurrent method used to compute the rows of the similarity matrix,
        # which contains the linear combination between the elements of the
        # similarity matrices based on meta-paths
        total = last-first
        start = timer()
        for i in range(first, last):
            values = [None]*len(w)
            values[0] = sparse.csr_matrix(self.Mapapa[i, :])
            values[1] = sparse.csr_matrix(self.May.getrow(i).dot(self.May.T))
            values[2] = sparse.csr_matrix(self.Mat.getrow(i).dot(self.Mat.T))
            denom = (self.Mav_diagonal[i]+self.Mav_diagonal).reshape((1,self.sims.shape[0]))
            num = 2*self.Mav.getrow(i).dot(self.Mav.T)
            values[3] = sparse.csr_matrix(num/denom)
            sims = np.zeros(self.sims.shape[0])
            for j in range(len(w)):
                sims += w[j]*values[j]
            semaphore.acquire()
            self.sims[i] = sparse.csr_matrix(np.where(sims>threshold, sims, zeros))
            semaphore.release()
            if(i%10==0):
                end = timer()
                elapsed = round(end-start,2)
                mins = math.floor(elapsed/60)
                secs = math.ceil(elapsed-mins*60)
                print("\rComputed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i-first,total,mins,secs),end='',flush=True)
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rComputed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i-first+1,total,mins,secs))

    def r_step(self, w:Iterable, n_workers:int=4, threshold:float=0.55, filename:str=None) -> None :
        """
        Computes the racall step (step 1): the Sim values are computed based on
        meta-paths AVA (same venue), APAPA (co-authors of my co-authors), APTPA
        (title similarities), APYPA (same publication year). The sum of
        similarity values of each meta-path is the resulting similarity score for
        each pair of author.

        Parameters
        ----------
        w : Iterable
            a vector with a weight for each similarity value based on a meta-path;
            the position are used as follows: { 0:APAPA, 1:APYPA, 2: APTPA, 3:AVA}
        n_workers : int, optional
            the number of concurrent workers used to compute the matrix (default
            is 4, but it is decreased to the number of cores of the machines if
            less than 4)
        threshold : float, optional
            used to determine wheter two authors are potential duplicates or not,
            i.e. if their Sim value is over this value (default is 0.55)
        filename : str, optional
            if it is not None, the matrix will be saved in the file with
            filename (default is None)

        Returns
        -------
        None
            the matrix is stored in the class attribute "self.sims"
        """
        if(self.Mapapa is None):
            raise NotImplementedError("can't compute p_step if Mapapa is None")
        if(self.May is None):
            raise NotImplementedError("can't compute p_step if May is None")
        if(self.Mat is None):
            raise NotImplementedError("can't compute p_step if Mat is None")
        if(self.Mav is None):
            raise NotImplementedError("can't compute p_step if Mav is None")
        if(self.Mav_diagonal is None):
            raise NotImplementedError("can't compute p_step if Mav_diagonal is None")
        if(not isinstance(w,Iterable)):
            raise TypeError("w must be an Iterable but %s was passed"%str(type(w)))
        if(not isinstance(n_workers,int)):
            raise TypeError("n_workers must be an int but %s was passed"%str(type(n_workers)))
        if(not isinstance(threshold,float)):
            raise TypeError("threshold must be a float but %s was passed"%str(type(threshold)))
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))
        if(n_workers<=0):
            raise ValueError("n_workers must be an int greater than 0 but %s was passed"%str(n_workers))
        if(threshold<=0.0 or threshold>1.0):
            raise ValueError("threshold must be a float greater than 0.0 and lower or equal to 1.0, but %s was passed"%str(threshold))

        zeros = np.zeros(self.Mav_diagonal.shape[0])
        self.sims = sparse.lil_matrix((self.Mav_diagonal.shape[0],self.Mav_diagonal.shape[0]))
        semaphore = Semaphore()
        n_workers = min(n_workers, multiprocessing.cpu_count())
        chunk = self.Mav_diagonal.shape[0]//n_workers
        workers = [None]*n_workers
        for i in range(n_workers):
            if(i<n_workers-1):
                workers[i] = Thread(target=self.__computeSims, args=(w, chunk*i, chunk*(i+1), zeros, semaphore, threshold))
            else:
                workers[i] = Thread(target=self.__computeSims, args=(w, chunk*i, self.Mav_diagonal.shape[0], zeros, semaphore, threshold))
            workers[i].start()
        for worker in workers:
            worker.join()
        if(filename is not None):
            sparse.save_npz(filename, self.sims.tocsr())

    def p_step(self, authors_dict:dict, filename:str=None) -> None :
        """
        Computes the precision step (step 2): for each author that has at least
        one possible duplicate from the recall step, compute the RatcliffObershelp
        similarity between his name and the name of his possible duplicates.

        Parameters
        ----------
        authors_dict : dict
            a dictionary where each pair should have an author's id as key and
            his name as value
        filename : str
            the file where the similarity values based on strings have to be stored

        Returns
        -------
        None
            the dictionary is stored in the class attribute "self.string_sims"
        """
        if(self.sims is None):
            raise NotImplementedError("can't compute r_step if sims is None")
        if(not isinstance(authors_dict,dict)):
            raise TypeError("authors_dict must be a dictionary but %s was passed"%str(type(authors_dict)))
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))

        self.string_sims = dict()
        ordinal_auth_ids = list(authors_dict.keys())
        start_time = timer()
        # for each author
        for i in range(self.sims.shape[0]):
            nnz_cols = self.sims[i].nonzero()[1]
            nnz_cols = np.delete(nnz_cols, np.where(nnz_cols==i))
            if(len(nnz_cols)>0):
                id0 = ordinal_auth_ids[i]
                auth_name = authors_dict[id0]
                sims = [None]*len(nnz_cols)
                ind = 0
                # for each of its resulting potential duplicates
                for j in nnz_cols:
                    id1 = ordinal_auth_ids[j]
                    sim = self.__ratcliff.normalized_similarity(auth_name, authors_dict[id1])
                    sims[ind] = (id1, sim)
                    ind += 1
                self.string_sims[id0] = sims
            if(i%10000==0):
                print("\rExamined: %d/%d"%(i,len(authors_dict)),end='',flush=True)
        end = timer()
        elapsed = round(end-start_time,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rExamined: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(i+1,len(authors_dict),mins,secs))
        if(filename is not None):
            with open(filename, 'w') as file:
                i = 0
                for key in self.string_sims:
                    file.write("%s;%s%s"%(key,json.dumps(self.string_sims[key]),("" if i==len(self.string_sims)-1 else "\n")))
                    i += 1

    def m_step(self, threshold:float=0.85, filename:str=None) -> None :
        """
        Computes the merge step (step 3): finding the actual duplicates for each
        authors pair that passed the first steps, i.e. the possible duplicate
        authors pairs with a similarity value (string based) over a threshold are
        considered to be actual duplicates.

        Parameters
        ----------
        threshold : float, optional
            used to determine wheter two authors are duplicates or not, i.e. if
            their RatcliffObershelp similarity value is over this value (default
            is 0.85)
        filename : str, optional
            if it is not None, the dictionary will be saved in the file with
            filename (default is None)

        Returns
        -------
        None
            the dictionary is stored in the class attribute "self.duplicates"
        """
        if(self.string_sims is None):
            raise NotImplementedError("can't compute the merge step if string_sims is None")
        if(not isinstance(threshold,float)):
            raise TypeError("threshold must be a float but %s was passed"%str(type(threshold)))
        if(filename is not None and not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))
        if(threshold<=0.0 or threshold>1.0):
            raise ValueError("threshold must be a float greater than 0.0 and lower or equal to 1.0, but %s was passed"%str(threshold))

        # Step 3.0: finding the actual duplicates for each author that passed
        # the first step, i.e. the possible duplicate authors with a similarity
        # value over a threshold
        self.duplicates = dict()
        j = 0
        start = timer()
        # for each author key from the list of keys with at least one possible
        # duplicate
        for key in self.string_sims:
            # for each value in the list of its possible duplicates
            for pair in self.string_sims[key]:
                # it is a confirmed duplicate
                if(pair[1]>=threshold):
                    # takes the key of the duplicate author
                    duplicate_key = pair[0]
                    if(not key in self.duplicates):
                        self.duplicates[key] = [duplicate_key]
                    else:
                        self.duplicates[key].append(duplicate_key)
            if(j%1000==0):
                print("\rComputed: %d/%d"%(j,len(self.string_sims)),end='',flush=True)
            j += 1
        end = timer()
        elapsed = round(end-start,2)
        mins = math.floor(elapsed/60)
        secs = math.ceil(elapsed-mins*60)
        print("\rComputed: %d/%d\t| Elapsed: %d min(s) %d sec(s)"%(j,len(self.string_sims),mins,secs))
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
            i = 0
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
                if(i%10==0):
                    print("\rKeys: %d/%d\t| Iterations: %d"%(i,len(current),iterations),end='',flush=True)
                i += 1
            print("\rKeys: %d/%d\t| Iterations: %d"%(i+1,len(current),iterations),end='',flush=True)
            iterations += 1
            current = next_dict
            next_dict = next_dict.copy()
        print("\rIterations: %d"%iterations)
        self.duplicates = next_dict
        if(filename is not None):
            with open(filename, 'w') as file:
                i = 0
                for key in self.duplicates:
                    file.write("%s;%s%s"%(key,self.duplicates[key],("" if i==len(self.duplicates)-1 else "\n")))
                    i += 1

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
            one (or more) from { Map,Mapapa,May,Mat,Mav,Mav_diagonal,Sims };
            other kwargs will be simply ignored without raising any error

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
                if(attr=="Mav_diagonal"):
                    self.Mav_diagonal = np.load(path)
                elif(attr=="Sims"):
                    self.sims = sparse.load_npz(path)
                elif(attr=="Map"):
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

    def loadStringSims(self, filename:str) -> None :
        """
        Loads the Sim values resulting from the recall step from file.

        Parameters
        ----------
        filename : str
            the file where the Sims values are stored

        Returns
        -------
        None
            the result is stored in the class attribute "self.string_sims"
        """
        if(not isinstance(filename,str)):
            raise TypeError("filename must be a string but %s was passed"%str(type(filename)))
        if(not os.path.exists(filename) or not os.path.isfile(filename)):
            raise ValueError("invalid filename")

        self.string_sims = dict()
        with open(filename, 'r') as file:
            line = file.readline()
            while(line!=''):
                line = line.rstrip()
                splt = line.split(";")
                self.string_sims[splt[0]] = json.loads(splt[1])
                line = file.readline()

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
